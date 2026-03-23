import os
import json
import time
import math
import argparse
import random
from dataclasses import asdict, dataclass
from typing import Tuple, Dict, Any, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from Nets import MLP, LeNetBN, ResNet18
from Binary_Nets import BinaryMLP, BinaryLeNetBN, BinaryResNet18


def setup_device(device_arg: str = "cuda"):
    """
    Decide and setup device.
    Assumes CUDA_VISIBLE_DEVICES is set externally if needed.
    """
    if device_arg.startswith("cuda") and torch.cuda.is_available():
        device = torch.device("cuda")
        torch.cuda.set_device(1)  # 在 CUDA_VISIBLE_DEVICES 语义下，始终用 cuda:0
    else:
        device = torch.device("cpu")

    return device

# -------------------------
# utils
# -------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # 更强可复现（会稍慢）
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def accuracy_from_log_probs(log_probs: torch.Tensor, y: torch.Tensor) -> float:
    pred = log_probs.argmax(dim=1)
    return (pred == y).float().mean().item()


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


@dataclass
class RunConfig:
    # meta
    dataset: str
    model: str
    method: str  # FP-Central / Binary-Central
    seed: int
    device: str

    # training
    epochs: int
    batch_size: int
    optimizer: str
    lr: float
    weight_decay: float
    momentum: float
    scheduler: str
    scheduler_step_size: int
    scheduler_gamma: float

    # misc (keep consistent with your paper config)
    binary_scale: bool = True
    binary_activation: bool = False
    encryption: str = "off"
    partition: str = "centralized"
    alpha: Optional[float] = None

    # data
    data_root: str = "./data"
    num_workers: int = 4
    pin_memory: bool = True

    # amp
    amp: bool = False


# -------------------------
# dataset loaders
# -------------------------
def get_datasets(dataset: str, data_root: str) -> Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset, int]:
    """
    Returns: train_set, test_set, num_classes
    Note:
      - MNIST/CIFAR10: torchvision ready
      - FEMNIST: 你如果用 LEAF 的 federated FEMNIST，这里需要你后面换成自己的 loader。
        这里先给一个可跑的 fallback：用 EMNIST(“byclass”) 近似替代，以便中心化流程先跑通。
    """
    dataset = dataset.lower()

    if dataset == "mnist":
        tfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        train = datasets.MNIST(root=data_root, train=True, download=True, transform=tfm)
        test = datasets.MNIST(root=data_root, train=False, download=True, transform=tfm)
        return train, test, 10

    if dataset in ["cifar10", "cifar-10"]:
        tfm_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        tfm_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        train = datasets.CIFAR10(root=data_root, train=True, download=False, transform=tfm_train)
        test = datasets.CIFAR10(root=data_root, train=False, download=False, transform=tfm_test)
        return train, test, 10

    if dataset in ["femnist", "emnist"]:
        # fallback: torchvision EMNIST as centralized stand-in
        tfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        train = datasets.EMNIST(root=data_root, split="byclass", train=True, download=True, transform=tfm)
        test = datasets.EMNIST(root=data_root, split="byclass", train=False, download=True, transform=tfm)
        num_classes = len(train.classes)
        return train, test, num_classes

    raise ValueError(f"Unknown dataset: {dataset}")


# -------------------------
# model factory
# -------------------------
def build_model(model_name: str, num_classes: int) -> nn.Module:
    m = model_name.lower()

    if m == "mlp":
        return MLP(num_classes=num_classes)
    if m == "binarymlp":
        return BinaryMLP(num_classes=num_classes)

    if m == "lenetbn":
        return LeNetBN(num_classes=num_classes)
    if m == "binarylenetbn":
        return BinaryLeNetBN(num_classes=num_classes)

    if m == "resnet18":
        return ResNet18(num_classes=num_classes)
    if m == "binaryresnet18":
        return BinaryResNet18(num_classes=num_classes)

    raise ValueError(f"Unknown model: {model_name}")


# -------------------------
# optim & sched
# -------------------------
def build_optimizer(cfg: RunConfig, model: nn.Module) -> torch.optim.Optimizer:
    name = cfg.optimizer.lower()
    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    if name == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=cfg.lr,
            momentum=cfg.momentum,
            weight_decay=cfg.weight_decay,
            nesterov=False,
        )
    raise ValueError(f"Unknown optimizer: {cfg.optimizer}")


def build_scheduler(cfg: RunConfig, optimizer: torch.optim.Optimizer):
    # name = cfg.scheduler.lower()
    # if name in ["none", ""]:
    #     return None
    # if name == "steplr":
    #     return torch.optim.lr_scheduler.StepLR(
    #         optimizer, step_size=cfg.scheduler_step_size, gamma=cfg.scheduler_gamma
    #     )
    # if name == "cosine":
    #     return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    # raise ValueError(f"Unknown scheduler: {cfg.scheduler}")
    return None

# -------------------------
# train/eval loops
# -------------------------
@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, criterion: nn.Module) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total = 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logp = model(x)
        loss = criterion(logp, y)

        total_loss += loss.item() * x.size(0)
        pred = logp.argmax(dim=1)
        total_correct += (pred == y).sum().item()
        total += x.size(0)

    return total_loss / max(total, 1), total_correct / max(total, 1)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    amp: bool,
) -> Tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total = 0

    scaler = torch.cuda.amp.GradScaler(enabled=(amp and device.type == "cuda"))

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=(amp and device.type == "cuda")):
            logp = model(x)
            loss = criterion(logp, y)

        if amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * x.size(0)
        pred = logp.argmax(dim=1)
        total_correct += (pred == y).sum().item()
        total += x.size(0)

    return total_loss / max(total, 1), total_correct / max(total, 1)

def PRINT_GPU_STATUS(device):
    print("=" * 60)
    print("[GPU INFO]")
    print(f"  CUDA_VISIBLE_DEVICES = {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not Set')}")
    print(f"  torch.cuda.is_available = {torch.cuda.is_available()}")
    print(f"  Using device = {device}")

    if device.type == "cuda":
        print(f"  GPU index = {torch.cuda.current_device()}")
        print(f"  GPU name  = {torch.cuda.get_device_name(0)}")
        print(f"  Total VRAM = {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print("=" * 60)


def _move_optimizer_state_to_device(optimizer: torch.optim.Optimizer, device: torch.device):
    # 断点续训常见坑：optimizer.state 里的 tensor 在 CPU，需要搬到 GPU
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)

def resume_from_ckpt(
    ckpt_path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
):
    print(f"[Resume] loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")

    # --- model ---
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.to(device)

    # --- optimizer ---
    optimizer.load_state_dict(ckpt["optimizer_state"])
    _move_optimizer_state_to_device(optimizer, device)

    # --- scheduler ---
    if scheduler is not None and ckpt.get("scheduler_state") is not None:
        scheduler.load_state_dict(ckpt["scheduler_state"])

    start_epoch = int(ckpt.get("epoch", 0)) + 1
    best_acc = float(ckpt.get("best_acc", -1.0))

    print(f"[Resume] start_epoch={start_epoch}, best_acc={best_acc:.6f}")
    return start_epoch, best_acc, ckpt
# -------------------------
# main
# -------------------------


def main():
    ap = argparse.ArgumentParser()

    # core
    ap.add_argument("--dataset", type=str, required=True, choices=["mnist", "femnist", "cifar10"])
    ap.add_argument("--model", type=str, required=True,
                    choices=["MLP", "BinaryMLP", "LeNetBN", "BinaryLeNetBN", "ResNet18", "BinaryResNet18"])
    ap.add_argument("--method", type=str, default=None,
                    help="Optional: override method name in config. If not set, inferred from model.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data_root", type=str, default="./data")

    # training
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--optimizer", type=str, default=None, choices=["Adam", "SGD"])
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--weight_decay", type=float, default=None)
    ap.add_argument("--momentum", type=float, default=0.9)

    # scheduler
    ap.add_argument("--scheduler", type=str, default=None, choices=["StepLR", "Cosine", "None"])
    ap.add_argument("--step_size", type=int, default=30)
    ap.add_argument("--gamma", type=float, default=0.5)

    # system
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--no_pin_memory", action="store_true")
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    # output
    ap.add_argument("--out_root", type=str, default="./runs_central")

    # ✅ 新增：resume
    ap.add_argument("--resume", type=str, default=None,
                    help="Path to ckpt_last.pt to resume training")

    args = ap.parse_args()

    set_seed(args.seed)
    device = setup_device(args.device)

    # infer defaults per dataset/model
    ds = args.dataset.lower()
    model_name = args.model

    if args.epochs is None:
        epochs = 150
    else:
        epochs = args.epochs

    if args.optimizer is None:
        if ds == "mnist":
            optimizer_name = "Adam"
        else:
            optimizer_name = "SGD"
    else:
        optimizer_name = args.optimizer

    if args.lr is None:
        if ds == "mnist" and optimizer_name.lower() == "adam":
            lr = 1e-3
        elif ds in ["femnist", "emnist"]:
            lr = 0.05
        else:
            lr = 0.1
    else:
        lr = args.lr

    if args.weight_decay is None:
        if optimizer_name.lower() == "adam":
            weight_decay = 0.0
        else:
            weight_decay = 5e-4
    else:
        weight_decay = args.weight_decay

    if args.scheduler is None:
        scheduler_name = "StepLR" if ds == "mnist" else "Cosine"
    else:
        scheduler_name = args.scheduler

    inferred_method = "Binary-Central" if "Binary" in model_name else "FP-Central"
    method = args.method if args.method is not None else inferred_method

    cfg = RunConfig(
        dataset=ds,
        model=model_name,
        method=method,
        seed=args.seed,
        device=str(device),
        epochs=epochs,
        batch_size=args.batch_size,
        optimizer=optimizer_name,
        lr=lr,
        weight_decay=weight_decay,
        momentum=args.momentum,
        scheduler=scheduler_name,
        scheduler_step_size=args.step_size,
        scheduler_gamma=args.gamma,
        data_root=args.data_root,
        num_workers=args.num_workers,
        pin_memory=(not args.no_pin_memory),
        amp=args.amp,
    )

    run_dir = os.path.join(args.out_root, cfg.dataset, cfg.model, cfg.method, f"seed_{cfg.seed}")
    ensure_dir(run_dir)

    # 保存 config（resume 时也会覆盖同名 config.json；通常没问题）
    with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2, ensure_ascii=False)

    # data
    train_set, test_set, num_classes = get_datasets(cfg.dataset, cfg.data_root)
    train_loader = DataLoader(
        train_set,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=(device.type == "cuda") and cfg.pin_memory,
        persistent_workers=(cfg.num_workers > 0),
    )
    test_loader = DataLoader(
        test_set,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=(device.type == "cuda") and cfg.pin_memory,
        persistent_workers=(cfg.num_workers > 0),
    )

    PRINT_GPU_STATUS(device)

    # model / loss / optim / sched
    model = build_model(cfg.model, num_classes=num_classes).to(device)
    criterion = nn.NLLLoss()
    optimizer = build_optimizer(cfg, model)
    scheduler = build_scheduler(cfg, optimizer)

    # metrics file
    metrics_path = os.path.join(run_dir, "metrics.csv")

    # ✅ 是否续训：决定 metrics 写入方式 和 start_epoch/best_acc
    start_epoch = 1
    best_acc = -1.0

    if args.resume is not None:
        # 续训：加载 ckpt，继续写 metrics（append，不重写表头）
        start_epoch, best_acc, ckpt = resume_from_ckpt(
            args.resume, model, optimizer, scheduler, device
        )
        if start_epoch > cfg.epochs:
            print(f"[Resume] checkpoint epoch ({start_epoch-1}) >= total epochs ({cfg.epochs}), nothing to do.")
            print(f"Done. Best test_acc = {best_acc*100:.2f}%. Saved to: {run_dir}")
            return
        if not os.path.exists(metrics_path):
            # 保险：如果 metrics 没了，就补一个表头
            with open(metrics_path, "w", encoding="utf-8") as f:
                f.write("round,train_loss,train_acc,test_loss,test_acc,lr,time_sec\n")
    else:
        # 新训练：重写 metrics 表头
        with open(metrics_path, "w", encoding="utf-8") as f:
            f.write("round,train_loss,train_acc,test_loss,test_acc,lr,time_sec\n")

    # ---- training loop ----
    for epoch in range(start_epoch, cfg.epochs + 1):
        t0 = time.time()
        if epoch < 50:
            optimizer.param_groups[0]["lr"] =0.01
        elif epoch < 100:
            optimizer.param_groups[0]["lr"] =0.005
        elif epoch < 85:
            optimizer.param_groups[0]["lr"] =0.001
        else: 
            optimizer.param_groups[0]["lr"] =0.0005
        train_loss, train_acc = train_one_epoch(
            model, train_loader, device, criterion, optimizer, amp=cfg.amp
        )
        test_loss, test_acc = evaluate(model, test_loader, device, criterion)

        if scheduler is not None:
            scheduler.step()

        lr_now = optimizer.param_groups[0]["lr"]
        dt = time.time() - t0

        # append metrics
        with open(metrics_path, "a", encoding="utf-8") as f:
            f.write(f"{epoch},{train_loss:.6f},{train_acc:.6f},{test_loss:.6f},{test_acc:.6f},{lr_now:.8f},{dt:.4f}\n")

        # checkpoints
        ckpt = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
            "best_acc": best_acc,
            "cfg": asdict(cfg),
        }
        torch.save(ckpt, os.path.join(run_dir, "ckpt_last.pt"))

        if test_acc > best_acc:
            best_acc = test_acc
            ckpt["best_acc"] = best_acc
            torch.save(ckpt, os.path.join(run_dir, "ckpt_best.pt"))

        print(f"[{cfg.dataset}/{cfg.model}/{cfg.method}][seed={cfg.seed}] "
              f"round {epoch:03d}/{cfg.epochs} | "
              f"train_acc={train_acc*100:.2f}% test_acc={test_acc*100:.2f}% | lr={lr_now:.3e} | {dt:.1f}s")

    print(f"Done. Best test_acc = {best_acc*100:.2f}%. Saved to: {run_dir}")

if __name__ == "__main__":
    main()
#python train_centralized.py --dataset femnist --model LeNetBN --method FP-Central005_100_decay --lr 0.05 --epochs 100
#python train_centralized.py --dataset femnist --model BinaryLeNetBN --method Binary-Central001_100_decay --lr 0.01 --epochs 100