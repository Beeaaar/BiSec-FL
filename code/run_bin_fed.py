from config import FLConfig_BinFL31,FLConfig_BinFL32,FLConfig_BinFL33,testConfig,FLConfig_BinFL_FEMNIST_20_5,FLConfig_BinFL_FEMNIST_100_10
from util import get_global_lr,build_model,set_seed,evaluate,train,binary_aggregate,get_device,sign_tensor
from data import get_datasets,partition_iid,partition_dirichlet,partition_femnist,partition_femnist_balanced
import torch
import csv
import numpy as np
from torch.utils.data import DataLoader, Subset
import os 
import sys
from binarized_modules import BinaryLinear,BinaryConv2d

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



def binarize_state_dict(full_state_dict, bin_param_names):
    """
    只对 bin_param_names 里的权重做 sign，其余保持 FP
    """
    out = {}
    bin_set = set(bin_param_names)
    for k, v in full_state_dict.items():
        if k in bin_set:
            out[k] = sign_tensor(v.to(dtype=torch.float32))
        else:
            out[k] = v.detach().clone()
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

    # ===== 初始化全局模型 =====
    global_model = build_model(cfg.model, num_classes).to(device)
    global_state = global_model.state_dict()
    bin_param_names, fp_param_names = find_binary_param_names(global_model)
    print("DEBUG,binary_parameter nums:",sum(p.numel() for name,p in global_model.named_parameters() if name in bin_param_names))
    print("DEBUG,fp_parameter nums",sum(p.numel() for name,p in global_model.named_parameters() if name in fp_param_names))
    global_hat = binarize_state_dict(global_state, bin_param_names)
    # ===== 记录文件（对齐旧集合版）=====
    run_dir = os.path.join(
        cfg.out_root,
        cfg.dataset,
        cfg.model,
        cfg.method,
        f"{cfg.partition}" + (f"_a{cfg.alpha}" if cfg.partition == "dirichlet" else ""),
        f"seed_{cfg.seed}",
    )
    os.makedirs(run_dir, exist_ok=True)

    metrics_path = os.path.join(run_dir, str(cfg.lr)+'_'+str(cfg.local_epochs)+'_'+str(cfg.num_clients)+'_'+str(cfg.clients_per_round)+"decay01-0001_metrics.csv")
    # 旧版：round,train_acc,test_acc,test_loss
    # 你想加时间就把 time 放进去
    with open(metrics_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["round", "train_acc", "test_acc", "test_loss"])

    history = []

    for r in range(cfg.rounds):
        #t0 = time.time()
        print(f"\n========== Round {r + 1}/{cfg.rounds} ==========")

        # ===== 1. 随机选择客户端 =====
        selected_clients = rng.choice(
            num_clients,
            size=cfg.clients_per_round,
            replace=False
        ).tolist()

        client_states = []
        client_sizes = []
        client_train_accs = []
        client_hat_states, nks = [], []
        # ===== 2. Client 本地训练 =====
        for k in selected_clients:
            local_model = build_model(cfg.model, num_classes).to(device)
            local_model.load_state_dict(global_hat, strict=True)   # Bin-FedAvg 下行发 global_hat

            subset = Subset(train_set, client_indices[k])
            cfg.lr = get_global_lr(r)
            train_acc,fp_state = train(local_model, subset, device, cfg)
            hat_state = binarize_state_dict(fp_state, bin_param_names)   # 上行也只二值化那部分

            client_hat_states.append(hat_state)
            nks.append(len(subset))


            client_train_accs.append(train_acc)

        mean_train_acc = float(np.mean(client_train_accs)) if client_train_accs else 0.0

        # ===== 3. Server 端 FedAvg =====
        global_hat = binary_aggregate(client_hat_states, nks, bin_param_names, fp_param_names)
        global_model.load_state_dict(global_hat, strict=True)

        # ===== 4. 全局评估 =====
        test_acc, test_loss = evaluate(
            global_model,
            test_loader,
            device
        )

        #dt = time.time() - t0

        # ===== 5. 写入 metrics.csv（对齐旧集合版字段顺序）=====
        with open(metrics_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([r + 1, mean_train_acc, test_acc, test_loss])

        history.append({
            "round": r + 1,
            "train_acc": mean_train_acc,
            "test_acc": test_acc,
            "test_loss": test_loss,
        })

        # 终端输出也对齐旧版风格
        print(
            f"[{cfg.dataset}/{cfg.model}/{cfg.method}][seed={cfg.seed}] "
            f"round {r+1:03d}/{cfg.rounds} | "
            f"train_acc={mean_train_acc*100:.2f}% "
            f"test_acc={test_acc*100:.2f}% "
        )

    return history

def main():
    #cfg = FLConfig_BinFL_FEMNIST_20_5()
    cfg = FLConfig_BinFL33()
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