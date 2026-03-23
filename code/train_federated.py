import os
import json
import math
import time
import copy
import random
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

# ====== 你的模型定义：直接 import 你已有的类 ======
# 你可以按你项目实际路径改一下 import
from Nets import MLP, LeNetBN, ResNet18
from Binary_Nets import BinaryMLP, BinaryLeNetBN, BinaryResNet18
from binarized_modules import BinaryLinear, BinaryConv2d


# -------------------------
# Config
# -------------------------
@dataclass
class FLConfig:
    # experiment identity
    dataset: str = "mnist"                # mnist | femnist | cifar10
    model: str = "MLP"                    # MLP | LeNetBN | ResNet18 | Binary*
    method: str = "fp_fedavg"             # fp_fedavg | bin_local_fedavg | bin_fedavg | cab_fl

    # data
    data_root: str = "./data"
    partition: str = "iid"                # iid | dirichlet
    alpha: float = 0.5                    # for dirichlet
    num_clients: int = 100
    seed: int = 0

    # federated
    rounds: int = 100
    clients_per_round: int = 10
    local_epochs: int = 1
    batch_size: int = 64

    # optimization
    optimizer: str = "sgd"                # sgd | adam
    lr: float = 0.05
    momentum: float = 0.9
    weight_decay: float = 5e-4

    # accel
    device: str = "cuda"
    num_workers: int = 4
    amp: bool = True

    # CAB-FL hyperparams
    beta0: float = 0.2
    cab_alpha: float = 1.0
    cab_gamma: float = 1.0

    # logging / output
    out_root: str = "./runs_fl"
    save_ckpt: bool = True
    debug:bool = True

# -------------------------
# Utils
# -------------------------



def get_device(device_arg: str) -> torch.device:
    if device_arg.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.set_device(0)  # under CUDA_VISIBLE_DEVICES
        return torch.device("cuda")
    return torch.device("cpu")


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def accuracy_from_logits(logits: torch.Tensor, y: torch.Tensor) -> float:
    pred = logits.argmax(dim=1)
    return (pred == y).float().mean().item()


@torch.no_grad()
def eval_model(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, float]:
    model.eval()
    total_acc = 0.0
    total_loss = 0.0
    n = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        loss = F.nll_loss(logits, y, reduction="sum")
        acc = (logits.argmax(dim=1) == y).sum()
        total_loss += loss.item()
        total_acc += acc.item()
        n += y.numel()
    return total_acc / max(n, 1), total_loss / max(n, 1)


# -------------------------
# Dataset
# -------------------------




# -------------------------
# Partitioning
# -------------------------


# -------------------------
# Binary param utilities
# -------------------------
def find_binary_param_names(model: nn.Module) -> Tuple[List[str], List[str], Dict[str, str]]:
    """
    Return (binary_param_names, fp_param_names, name_to_layerkey)
    layerkey is a coarse "layer id" used for confidence calculation.
    """
    bin_names = []
    fp_names = []
    name_to_layerkey = {}

    # map module prefix -> layerkey
    for mod_name, mod in model.named_modules():
        if isinstance(mod, (BinaryLinear, BinaryConv2d)):
            # Usually binary weights are under "<mod_name>.weight"
            w_name = f"{mod_name}.weight"
            bin_names.append(w_name)
            name_to_layerkey[w_name] = mod_name  # treat module name as layer id

    # classify remaining params as FP
    all_param_names = [n for n, _ in model.named_parameters()]
    bin_set = set(bin_names)
    for n in all_param_names:
        if n in bin_set:
            continue
        fp_names.append(n)
        name_to_layerkey[n] = n.split(".")[0]  # coarse

    return bin_names, fp_names, name_to_layerkey


def sign_tensor(x: torch.Tensor) -> torch.Tensor:
    # deterministic sign; map zeros to +1
    s = torch.sign(x)
    s[s == 0] = 1
    return s


def state_dict_clone(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {k: v.detach().clone() for k, v in sd.items()}


def weighted_average(states: List[Dict[str, torch.Tensor]], weights: List[float]) -> Dict[str, torch.Tensor]:
    out = {}
    for k in states[0].keys():
        acc = None
        for s, w in zip(states, weights):
            t = s[k].to(dtype=torch.float32)
            acc = t * w if acc is None else acc + t * w
        out[k] = acc
    return out


# -------------------------
# Local training (client)
# -------------------------
def build_optimizer(cfg: FLConfig, model: nn.Module) -> torch.optim.Optimizer:
    if cfg.optimizer.lower() == "adam":
        return torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    return torch.optim.SGD(model.parameters(), lr=cfg.lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay)


def local_train_one_client(
    cfg: FLConfig,
    model: nn.Module,
    train_subset: Subset,
    device: torch.device,
) -> Tuple[float, Dict[str, torch.Tensor]]:

    # ---- helper: BN-safe switch ----
    def set_bn_eval(m: nn.Module):
        for mod in m.modules():
            if isinstance(mod, (nn.BatchNorm1d, nn.BatchNorm2d)):
                mod.eval()

    def set_bn_train(m: nn.Module):
        for mod in m.modules():
            if isinstance(mod, (nn.BatchNorm1d, nn.BatchNorm2d)):
                mod.train()

    model.train()

    loader = DataLoader(
        train_subset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(cfg.num_workers > 0),
    )

    opt = build_optimizer(cfg, model)
    scaler = torch.cuda.amp.GradScaler(enabled=(cfg.amp and device.type == "cuda"))

    total = 0
    correct = 0

    for _ in range(cfg.local_epochs):
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            # ===== BN-safe handling =====
            if x.size(0) < 2:
                set_bn_eval(model)    # freeze BN for tiny batch
            else:
                set_bn_train(model)  # normal BN training

            opt.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=(cfg.amp and device.type == "cuda")):
                logits = model(x)
                loss = F.nll_loss(logits, y)

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            total += y.numel()
            correct += (logits.argmax(dim=1) == y).sum().item()

    train_acc = correct / max(total, 1)

    # IMPORTANT: clone state_dict to avoid reference sharing
    return train_acc, state_dict_clone(model.state_dict())


# -------------------------
# CAB-FL client-side fusion
# -------------------------
@torch.no_grad()
def cab_fuse_into_shadow(
    model: nn.Module,
    global_hat: Dict[str, torch.Tensor],
    global_conf: Dict[str, float],   # layer_key -> c_g
    cfg: FLConfig,
    bin_param_names: List[str],
    name_to_layerkey: Dict[str, str],
):
    """
    Layer-wise Confidence-Aware fusion (CAB-FL).

    Steps:
    1) Compute layer-level local confidence c_k(l)
    2) Compute beta_l for each layer
    3) Apply fusion to all binary params in that layer
    """

    device = next(model.parameters()).device
    sd = model.state_dict()

    # -------------------------------------------------
    # 1. Compute layer-level local confidence c_k(l)
    # -------------------------------------------------
    layer_matches = {}  # layer_key -> list of match ratios

    for n in bin_param_names:
        layer_key = name_to_layerkey[n]

        w = sd[n].to(device=device, dtype=torch.float32)
        hat = global_hat[n].to(device=device, dtype=torch.float32)

        match = (sign_tensor(w) == hat).float().mean().item()
        layer_matches.setdefault(layer_key, []).append(match)

    # mean over parameters in the same layer
    layer_ck = {
        lk: float(np.mean(v)) for lk, v in layer_matches.items()
    }

    # -------------------------------------------------
    # 2. Compute layer-wise beta
    # -------------------------------------------------
    layer_beta = {}
    for lk, c_k in layer_ck.items():
        c_g = float(global_conf.get(lk, 0.0))
        beta = cfg.beta0 * (c_g ** cfg.cab_alpha) * (c_k ** cfg.cab_gamma)
        beta = float(np.clip(beta, 0.0, 1.0))
        layer_beta[lk] = beta

        if getattr(cfg, "debug", False):
            print(
                f"[DEBUG][CAB] layer={lk} "
                f"c_g={c_g:.4f} c_k={c_k:.4f} beta={beta:.6f}"
            )

    # -------------------------------------------------
    # 3. Apply fusion (shared beta per layer)
    # -------------------------------------------------
    for n in bin_param_names:
        layer_key = name_to_layerkey[n]
        beta = layer_beta[layer_key]

        if beta <= 0.0:
            continue  # skip useless fusion

        w = sd[n].to(device=device, dtype=torch.float32)
        hat = global_hat[n].to(device=device, dtype=torch.float32)

        fused = (1.0 - beta) * w + beta * hat
        sd[n].copy_(fused.to(dtype=sd[n].dtype))

    model.load_state_dict(sd, strict=True)



# -------------------------
# Server aggregation
# -------------------------
def aggregate_fp_fedavg(client_states: List[Dict[str, torch.Tensor]], nks: List[int]) -> Dict[str, torch.Tensor]:
    weights = [nk / max(sum(nks), 1) for nk in nks]
    return weighted_average(client_states, weights)

def aggregate_binary_fedavg(
    client_hat_states: List[Dict[str, torch.Tensor]],
    nks: List[int],
    bin_param_names: List[str],
    fp_param_names: List[str],
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """
    Binary-FedAvg aggregation.

    - Binary params: weighted average in float -> sign
    - FP params: standard FedAvg
    - BN buffers and other params: copied from template
    """
    weights = [nk / max(sum(nks), 1) for nk in nks]

    # ===== use full state_dict as template =====
    template = client_hat_states[0]
    global_hat = {k: v.clone() for k, v in template.items()}

    tilde = {}

    # ===== binary params =====
    for n in bin_param_names:
        acc = None
        for s, w in zip(client_hat_states, weights):
            t = s[n].to(dtype=torch.float32)
            acc = t * w if acc is None else acc + t * w
        tilde[n] = acc
        global_hat[n] = acc.sign()

    # ===== fp params =====
    for n in fp_param_names:
        acc = None
        for s, w in zip(client_hat_states, weights):
            t = s[n].to(dtype=torch.float32)
            acc = t * w if acc is None else acc + t * w
        global_hat[n] = acc

    return global_hat, tilde



def compute_global_confidence(
    client_hat_states: List[Dict[str, torch.Tensor]],
    bin_param_names: List[str],
    name_to_layerkey: Dict[str, str],
) -> Dict[str, float]:
    layer_sum, layer_cnt = {}, {}
    K = len(client_hat_states)

    for n in bin_param_names:
        layer = name_to_layerkey[n]
        signs = torch.stack([s[n].float() for s in client_hat_states], dim=0)  # [K, ...], values in {-1,+1}

        # p = fraction of +1
        p = (signs > 0).float().mean(dim=0)          # [...]
        disagree = 4.0 * p * (1.0 - p)               # in [0,1]
        conf = 1.0 - disagree                        # in [0,1]
        v = conf.mean().item()

        layer_sum[layer] = layer_sum.get(layer, 0.0) + v
        layer_cnt[layer] = layer_cnt.get(layer, 0) + 1

    return {l: layer_sum[l] / max(layer_cnt[l], 1) for l in layer_sum}




def avg_confidence_scalar(global_conf: Dict[str, float]) -> float:
    if not global_conf:
        return 0.0
    return float(sum(global_conf.values()) / len(global_conf))

def binarize_state_dict(full_state_dict, bin_param_names):
    bin_state = {}
    for k, v in full_state_dict.items():
        #print(k)
        if k in bin_param_names:
            # only binarize trainable binary parameters
            bin_state[k] = v.sign()
        else:
            # keep everything else (BN buffers, FP layers, etc.)
            bin_state[k] = v.clone()
    return bin_state
# -------------------------
# Main FL loop
# -------------------------
def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, default="mnist")
    ap.add_argument("--model", type=str, default="MLP")
    ap.add_argument("--method", type=str, default="fp_fedavg",
                    choices=["fp_fedavg", "bin_local_fedavg", "bin_fedavg", "cab_fl"])
    ap.add_argument("--partition", type=str, default="iid", choices=["iid", "dirichlet"])
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--num_clients", type=int, default=100)
    ap.add_argument("--clients_per_round", type=int, default=10)
    ap.add_argument("--rounds", type=int, default=100)
    ap.add_argument("--local_epochs", type=int, default=1)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--optimizer", type=str, default="sgd", choices=["sgd", "adam"])
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--momentum", type=float, default=0.9)
    ap.add_argument("--weight_decay", type=float, default=5"./data"e-4)
    ap.add_argument("--data_root", type=str, default="./data")
    ap.add_argument("--out_root", type=str, default="./runs_fl")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--no_amp", action="store_true")
    # CAB-FL params
    ap.add_argument("--beta0", type=float, default=0.2)
    ap.add_argument("--cab_alpha", type=float, default=1.0)
    ap.add_argument("--cab_gamma", type=float, default=1.0)
    args = ap.parse_args()

    cfg = FLConfig(
        dataset=args.dataset,
        model=args.model,
        method=args.method,
        partition=args.partition,
        alpha=args.alpha,
        num_clients=args.num_clients,
        clients_per_round=args.clients_per_round,
        rounds=args.rounds,
        local_epochs=args.local_epochs,
        batch_size=args.batch_size,
        optimizer=args.optimizer,
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        data_root=args.data_root,
        out_root=args.out_root,
        seed=args.seed,
        num_workers=args.num_workers,
        amp=(not args.no_amp),
        beta0=args.beta0,
        cab_alpha=args.cab_alpha,
        cab_gamma=args.cab_gamma,
        debug = True,
    )

    set_seed(cfg.seed)
    device = get_device(cfg.device)

    print("=" * 70)
    print("[GPU INFO]")
    print(f"  CUDA_VISIBLE_DEVICES = {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not Set')}")
    print(f"  torch.cuda.is_available = {torch.cuda.is_available()}")
    print(f"  Using device = {device}")
    if device.type == "cuda":
        print(f"  GPU index = {torch.cuda.current_device()}")
        print(f"  GPU name  = {torch.cuda.get_device_name(0)}")
    print("=" * 70)

    train_set, test_set, num_classes = get_datasets(cfg.dataset, cfg.data_root)


    # global model
    global_model = build_model(cfg.model, num_classes=num_classes).to(device)
    bin_param_names, fp_param_names, name_to_layerkey = find_binary_param_names(global_model)

    # output dir
    run_dir = os.path.join(
        cfg.out_root,
        cfg.dataset,
        cfg.model,
        cfg.method,
        f"{cfg.partition}" + (f"_a{cfg.alpha}" if cfg.partition == "dirichlet" else ""),
        f"seed_{cfg.seed}",
    )
    ensure_dir(run_dir)

    with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2)

    metrics_path = os.path.join(run_dir, "metrics.csv")
    with open(metrics_path, "w", encoding="utf-8") as f:
        cols = ["round", "train_acc", "test_acc", "test_loss"]
        if cfg.method == "cab_fl":
            cols.append("avg_confidence")
        f.write(",".join(cols) + "\n")

    test_loader = DataLoader(
        test_set,
        batch_size=256,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(cfg.num_workers > 0),
    )

    # initialize global state
    global_state = state_dict_clone(global_model.state_dict())
    global_hat = None
    global_conf = None
    
    if cfg.method in ["bin_fedavg", "cab_fl"]:
        global_hat = binarize_state_dict(global_state, bin_param_names)
    for t in range(cfg.rounds):
        t0 = time.time()
        # sample clients
        rng = np.random.default_rng(cfg.seed + t)
        selected = rng.choice(cfg.num_clients, size=cfg.clients_per_round, replace=False).tolist()

        client_states = []
        client_nks = []
        train_accs = []

        # broadcast + local update
        for k in selected:
            # create local model and load appropriate state
            local_model = build_model(cfg.model, num_classes=num_classes).to(device)

            # choose what to receive
            if cfg.method in ["fp_fedavg", "bin_local_fedavg"]:
                local_model.load_state_dict(global_state, strict=True)
            else:
                # binary aggregation methods receive global_hat
                assert global_hat is not None, "global_hat should be initialized after first aggregation."
                local_model.load_state_dict(global_hat, strict=True)
                if cfg.method == "cab_fl" and global_conf is not None:
                    cab_fuse_into_shadow(
                        local_model, global_hat, global_conf, cfg, bin_param_names, name_to_layerkey
                    )

            subset = Subset(train_set, client_idxs[k])
            nk = len(client_idxs[k])

            train_acc, new_state = local_train_one_client(cfg, local_model, subset, device)
            train_accs.append(train_acc)
            client_nks.append(nk)

            if cfg.method == "fp_fedavg":
                client_states.append(new_state)

            elif cfg.method == "bin_local_fedavg":
                # local binary training, but upload FP shadow (the raw state_dict)
                client_states.append(new_state)

            elif cfg.method in ["bin_fedavg", "cab_fl"]:
                # upload hat_w for binary params, and FP for fp params
                sd_hat = {}
                for n, v in new_state.items():
                    if n in bin_param_names:
                        sd_hat[n] = sign_tensor(v.to(dtype=torch.float32)).cpu()
                    else:
                        sd_hat[n] = v.to(dtype=torch.float32).cpu()
                client_states.append(sd_hat)

            else:
                raise ValueError(f"Unknown method: {cfg.method}")

        # server aggregation
        if cfg.method in ["fp_fedavg", "bin_local_fedavg"]:
            global_state = aggregate_fp_fedavg(client_states, client_nks)
            # for binary local baseline, keep global_state as FP; clients will binarize internally via Binary modules
            # (your BinaryLinear/BinaryConv2d forward handles it)
        else:
            global_hat, tilde = aggregate_binary_fedavg(
                client_states, client_nks, bin_param_names, fp_param_names
            )

            global_conf = compute_global_confidence(
                client_states,
                bin_param_names,
                name_to_layerkey,
            )


        # evaluate global model
        if cfg.method in ["fp_fedavg", "bin_local_fedavg"]:
            global_model.load_state_dict(global_state, strict=True)
        else:
            #gm_keys = set(global_model.state_dict().keys())
            #gh_keys = set(global_hat.keys())

            #missing = sorted(list(gm_keys - gh_keys))
            #extra   = sorted(list(gh_keys - gm_keys))

            #print("[DEBUG] global_hat keys:", len(gh_keys), "global_model keys:", len(gm_keys))
            #print("[DEBUG] missing in global_hat (first 20):", missing[:20])
            #print("[DEBUG] has bn1.running_mean?", "bn1.running_mean" in gh_keys)
            #print("[DEBUG] has bn1.num_batches_tracked?", "bn1.num_batches_tracked" in gh_keys)

            global_model.load_state_dict(global_hat, strict=True)

        test_acc, test_loss = eval_model(global_model, test_loader, device)

        mean_train_acc = float(np.mean(train_accs)) if train_accs else 0.0
        line = [t + 1, mean_train_acc, test_acc, test_loss]
        if cfg.method == "cab_fl":
            line.append(avg_confidence_scalar(global_conf))
        with open(metrics_path, "a", encoding="utf-8") as f:
            f.write(",".join([str(x) for x in line]) + "\n")

        # save checkpoint
        if cfg.save_ckpt and ((t + 1) % 20 == 0 or (t + 1) == cfg.rounds):
            ckpt = {
                "round": t + 1,
                "cfg": asdict(cfg),
                "global_state": global_state if cfg.method in ["fp_fedavg", "bin_local_fedavg"] else None,
                "global_hat": global_hat if cfg.method in ["bin_fedavg", "cab_fl"] else None,
                "global_conf": global_conf if cfg.method == "cab_fl" else None,
            }
            torch.save(ckpt, os.path.join(run_dir, f"ckpt_round_{t+1}.pt"))

        dt = time.time() - t0
        if cfg.method == "cab_fl":
            print(f"[{cfg.dataset}/{cfg.model}/{cfg.method}][seed={cfg.seed}] "
                  f"round {t+1:03d}/{cfg.rounds} | train_acc={mean_train_acc*100:.2f}% "
                  f"test_acc={test_acc*100:.2f}% | avg_conf={avg_confidence_scalar(global_conf):.4f} | {dt:.1f}s")
        else:
            print(f"[{cfg.dataset}/{cfg.model}/{cfg.method}][seed={cfg.seed}] "
                  f"round {t+1:03d}/{cfg.rounds} | train_acc={mean_train_acc*100:.2f}% "
                  f"test_acc={test_acc*100:.2f}% | {dt:.1f}s")


if __name__ == "__main__":
    main()
