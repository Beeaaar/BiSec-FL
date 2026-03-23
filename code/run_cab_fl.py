from config import FLConfig_CabFL41,FLConfig_CabFL42,FLConfig_CabFL43,FLConfig_CabFL_FEMNIST_20_5,FLConfig_CabFL_FEMNIST_100_10
from util import get_global_lr,build_model,set_seed,evaluate,train,aggregate_bin_sign_and_fp,aggregate_scales,get_device,sign_tensor
from data import get_datasets,partition_iid,partition_dirichlet,partition_femnist,partition_femnist_balanced
import torch
import csv
import numpy as np
from torch.utils.data import DataLoader, Subset
import os 
import sys
from binarized_modules import BinaryLinear,BinaryConv2d
import math

def log_binary_scaling_factors(model):
    stats = []
    for name, module in model.named_modules():
        if isinstance(module, (BinaryConv2d, BinaryLinear)):
            with torch.no_grad():
                w = module.weight
                alpha = w.abs().mean().item()
                stats.append((name, alpha))
    return stats


def find_binary_param_names(model):
    """
    返回：bin_param_names, fp_param_names
    bin_param_names：所有 BinaryLinear/BinaryConv2d 的 weight 参数名
    fp_param_names：其余所有“可训练参数名”（包括最后一层、bias、BN的weight/bias等）
    """
    bin_param_names = []
    for mod_name, mod in model.named_modules():
        if isinstance(mod, (BinaryLinear, BinaryConv2d)):
            bin_param_names.append(f"{mod_name}.weight")

    all_param_names = [n for n, _ in model.named_parameters()]
    bin_set = set(bin_param_names)
    fp_param_names = [n for n in all_param_names if n not in bin_set]
    return bin_param_names, fp_param_names

def compute_client_scale(fp_state, bin_param_names, eps=1e-12):
    """
    返回 dict[name] = scalar scale_k
    scale_k 用 mean(|w_fp|) 表示该二值权重张量的幅值尺度
    """
    sk = {}
    for name in bin_param_names:
        w = fp_state[name].detach().float()
        sk[name] = float(w.abs().mean().item() + eps)
    return sk


    
@torch.no_grad()
def apply_residual_compensation_and_pack(
    tilde_bin,              # dict[name] float tensor in [-1,1]
    agg_scale,              # dict[name] scalar
    residual_buf,           # dict[name] float tensor (same shape as weight)
    mu=0.9,                 # residual momentum
    lam=1e-2,               # residual strength
):
    """
    1) pre = tilde + lam * residual
    2) new_sign = sign(pre)
    3) residual = mu*residual + (tilde - new_sign)
    4) pack global binary weight = new_sign * scale
    """
    global_bin = {}

    for name, tb in tilde_bin.items():
        if name not in residual_buf or residual_buf[name].shape != tb.shape:
            residual_buf[name] = torch.zeros_like(tb, dtype=torch.float32)

        pre = tb.to(torch.float32) + lam * residual_buf[name]
        new_sign = sign_tensor(pre)

        # error-feedback style update
        residual_buf[name] = mu * residual_buf[name] + (tb.to(torch.float32) - new_sign)

        scale = float(agg_scale[name])
        global_bin[name] = new_sign * scale  # store as sign*scale

    return global_bin, residual_buf

@torch.no_grad()
def compute_client_confidence_delta(fp_state, bin_param_names, eps=1e-8, beta=50.0):
    """
    Δ-aware client confidence:
    ck[name] = exp(-beta * mean(|w - sign(w)*alpha|)/(alpha+eps))
    where alpha = mean(|w|)
    """
    ck = {}
    for name in bin_param_names:
        w = fp_state[name].detach().float()
        alpha = w.abs().mean()
        w_bin = sign_tensor(w) * alpha
        delta = (w - w_bin).abs().mean() / (alpha + eps)   # normalized deviation
        ck[name] = math.exp(-beta * float(delta))
    return ck

def binarize_state_dict_with_scale(full_state_dict, bin_param_names, eps=1e-12):
    """
    binary param: sign(w) * scale, where scale = mean(|w|)
    fp/BN/bias: keep FP
    """
    out = {}
    bin_set = set(bin_param_names)
    for k, v in full_state_dict.items():
        if k in bin_set:
            w = v.detach().float()
            scale = float(w.abs().mean().item() + eps)
            out[k] = sign_tensor(w) * scale
        else:
            out[k] = v.detach().clone()
    return out

def gate_and_update(prev_hat, new_hat, cg, avg_ck, bin_param_names):
    out = {}
    bin_set = set(bin_param_names)

    for k in new_hat.keys():
        if k in bin_set:
            eta = 0.5 * (cg[k] + avg_ck[k])      # simple gate
            if eta < 0.0: eta = 0.0
            if eta > 1.0: eta = 1.0

            # soft update then sign
            out[k] = sign_tensor((1 - eta) * prev_hat[k].float() + eta * new_hat[k].float())
        else:
            out[k] = new_hat[k]
    return out

def gate_and_update_delta(prev_hat, new_hat, cg, bin_param_names, eta_min=0.0, eta_max=1.0):
    """
    Δ-aware gate:
    eta_l = clip(cg_l, [eta_min, eta_max])  # simplest: directly use cg as gate strength
    binary param: sign((1-eta)*prev + eta*new)
    fp params: keep new_hat
    """
    bin_set = set(bin_param_names)
    out = {}

    for k in new_hat.keys():
        if k in bin_set:
            eta = float(cg[k])
            if eta < eta_min: eta = eta_min
            if eta > eta_max: eta = eta_max
            out[k] = sign_tensor((1 - eta) * prev_hat[k].float() + eta * new_hat[k].float())
        else:
            out[k] = new_hat[k]
    return out

def fl_round_loop(
    cfg,
    train_set,
    test_loader,
    client_indices,
    num_classes,
    device,
):
    num_clients = cfg.num_clients
    rng = np.random.default_rng(cfg.seed)

    # ===== 0. 初始化全局模型 =====
    global_model = build_model(cfg.model, num_classes).to(device)
    global_state = global_model.state_dict()

    bin_param_names, fp_param_names = find_binary_param_names(global_model)

    # global_hat: binary weights stored as (sign * scale)
    global_hat = binarize_state_dict_with_scale(global_state, bin_param_names)

    # server-side residual buffer (no communication)
    residual_buf = {}

    # ===== 日志目录 =====
    run_dir = os.path.join(
        cfg.out_root,
        cfg.dataset,
        cfg.model,
        cfg.method,
        f"{cfg.partition}" + (f"_a{cfg.alpha}" if cfg.partition == "dirichlet" else ""),
        f"seed_{cfg.seed}",
    )
    os.makedirs(run_dir, exist_ok=True)

    metrics_path = os.path.join(run_dir, str(cfg.lr)+'_'+str(cfg.local_epochs)+'_'+str(cfg.num_clients)+'_'+str(cfg.clients_per_round)+"mu0.4lam1-4_metrics.csv")
    with open(metrics_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["round", "train_acc", "test_acc", "test_loss"])

    history = []

    # ============================================================
    # ===================== FL Main Loop =========================
    # ============================================================
    for r in range(cfg.rounds):
        print(f"\n========== Round {r + 1}/{cfg.rounds} ==========")

        # ===== 1. 采样客户端 =====
        selected_clients = rng.choice(
            num_clients,
            size=cfg.clients_per_round,
            replace=False
        ).tolist()

        client_hat_states = []     # uplink: binary sign + fp params
        client_scale_list = []     # uplink: per-binary-weight scalar
        nks = []
        client_train_accs = []

        # ===== 2. Client 本地训练 =====
        for k in selected_clients:
            local_model = build_model(cfg.model, num_classes).to(device)

            # 下发 global_hat（binary 已是 sign*scale）
            local_model.load_state_dict(global_hat, strict=True)

            subset = Subset(train_set, client_indices[k])
            #cfg.lr = get_global_lr(r)
            train_acc, fp_state = train(local_model, subset, device, cfg)

            # ---- 上传 scale（每个 binary weight 一个标量）----
            sk = compute_client_scale(fp_state, bin_param_names)
            client_scale_list.append(sk)

            #---- 上传 binary sign + fp params ----
            hat_state = {}
            bin_set = set(bin_param_names)
            for name, v in fp_state.items():
                if name in bin_set:
                    hat_state[name] = sign_tensor(v.detach().float()).cpu()
                else:
                    hat_state[name] = v.detach().float().cpu()

            #========= Client uplink: sign(delta) =========


            client_hat_states.append(hat_state)
            nks.append(len(subset))
            client_train_accs.append(train_acc)

        mean_train_acc = float(np.mean(client_train_accs)) if client_train_accs else 0.0

        # ===== 3. Server 聚合 =====

        # (a) sign 聚合（得到 tilde_bin）+ FP FedAvg
        tilde_bin, agg_fp = aggregate_bin_sign_and_fp(
            client_hat_states,
            nks,
            bin_param_names,
            fp_param_names,
        )

        # (b) scale 聚合（每个 binary weight 一个标量）
        agg_scale = aggregate_scales(
            client_scale_list,
            nks,
            bin_param_names,
            scale_min=getattr(cfg, "scale_min", 1e-6),
            scale_max=getattr(cfg, "scale_max", 1e6),
        )
        #默认0.9,1e-2
        mu = 0.4
        lam = 1e-4
        # (c) residual compensation + pack sign*scale
        global_bin, residual_buf = apply_residual_compensation_and_pack(
            tilde_bin=tilde_bin,
            agg_scale=agg_scale,
            residual_buf=residual_buf,
            mu=getattr(cfg, "res_mu", mu),
            lam=getattr(cfg, "res_lam", lam),
        )

        # (d) 组装新的 global_hat
        global_hat = {}
        for name in agg_fp.keys():
            global_hat[name] = agg_fp[name]
        for name in bin_param_names:
            global_hat[name] = global_bin[name]

        global_model.load_state_dict(global_hat, strict=True)
        # 在 server update 后


        # stats = log_binary_scaling_factors(global_model)
        # for name, alpha in stats:
        #     if "layer4" in name or "shortcut" in name:
        #         print(f"[DEBUG] {name}: alpha = {alpha:.6f}")
        # ===== 4. 全局评估 =====
        test_acc, test_loss = evaluate(
            global_model,
            test_loader,
            device
        )

        # ===== 5. 写日志 =====
        with open(metrics_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([r + 1, mean_train_acc, test_acc, test_loss])

        history.append({
            "round": r + 1,
            "train_acc": mean_train_acc,
            "test_acc": test_acc,
            "test_loss": test_loss,
        })

        print(
            f"[{cfg.dataset}/{cfg.model}/{cfg.method}][seed={cfg.seed}] "
            f"round {r+1:03d}/{cfg.rounds} | "
            f"train_acc={mean_train_acc*100:.2f}% "
            f"test_acc={test_acc*100:.2f}%"
        )

    return history
def main():
    #cfg = FLConfig_CabFL_FEMNIST_20_5()
    cfg = FLConfig_CabFL42()
    #cfg = FLConfig_CabFL43()
    set_seed(cfg.seed)

    device = get_device(cfg.device)
    train_set, test_set, num_classes = get_datasets(cfg.dataset, cfg.data_root)
    #print(train_set,test_set,num_classes)
    test_loader = DataLoader(
        test_set,
        batch_size=256,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(cfg.num_workers > 0),
    )

    if cfg.partition == "iid":
        client_idxs = partition_iid(train_set, cfg.num_clients, cfg.seed)
    elif cfg.partition == "femnist":
        client_idxs = partition_femnist_balanced(
            train_set,
            num_clients=cfg.num_clients,
            seed=cfg.seed
        )
    else:
        client_idxs = partition_dirichlet(train_set, cfg.num_clients, cfg.alpha, cfg.seed)


    history = fl_round_loop(cfg,train_set,test_loader,client_idxs,num_classes,device)    


if __name__ == '__main__':
    main()