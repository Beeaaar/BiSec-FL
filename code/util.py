import torch
import random
import numpy as np
from Nets import MLP,LeNetBN,ResNet18
from Binary_Nets import BinaryMLP,BinaryLeNetBN,BinaryResNet18
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
from torchvision import datasets, transforms
import math

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def sign_tensor(x: torch.Tensor) -> torch.Tensor:
    s = torch.sign(x)
    s[s == 0] = 1
    return s

def clone_state(sd):
    return {k: v.detach().clone() for k, v in sd.items()}

def build_model(model_name: str, num_classes: int) -> nn.Module:
    m = model_name.lower()
    if m == "mlp":
        return MLP(num_classes=num_classes)
    if m == "lenetbn":
        return LeNetBN(num_classes=num_classes)
    if m == "resnet18":
        return ResNet18(num_classes=num_classes)

    if m == "binarymlp":
        return BinaryMLP(num_classes=num_classes)
    if m == "binarylenetbn":
        return BinaryLeNetBN(num_classes=num_classes)
    if m == "binaryresnet18":
        return BinaryResNet18(num_classes=num_classes)

    raise ValueError(f"Unknown model: {model_name}")

def get_device(device_arg: str) -> torch.device:
    if device_arg.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.set_device(3)  # under CUDA_VISIBLE_DEVICES
        return torch.device("cuda")
    return torch.device("cpu")

def train(model, train_subset, device, cfg):
    """
    Full-Precision local training for one client.

    Semantic guarantees:
    - FP weights
    - FP forward
    - FP backward
    - Upload FP shadow parameters
    """

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
        drop_last=False,
    )

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=cfg.lr,
        momentum=getattr(cfg, "momentum", 0.0),
        weight_decay=getattr(cfg, "weight_decay", 0.0),
    )
    total = 0
    correct = 0
    for _ in range(cfg.local_epochs):
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            if x.size(0) < 2:
                set_bn_eval(model)    # freeze BN for tiny batch
            else:
                set_bn_train(model)  # normal BN training
            optimizer.zero_grad()
            logits = model(x)             
            loss = F.nll_loss(logits, y)
            loss.backward()                   
            optimizer.step()                 
            
            total += y.numel()
            correct += (logits.argmax(dim=1) == y).sum().item()

    train_acc = correct / max(total, 1)

    # 返回的是 FP shadow
    return train_acc, clone_state(model.state_dict())

def get_global_lr(round_idx: int):
    if round_idx < 230:
        return 0.07-round_idx*(0.07-0.001)/230
    else:
        return 0.001
    #return 0.05-round_idx*(0.05-0.005)/250
    # if round_idx <= 30:
    #     return 0.1
    # elif round_idx <= 0:
    #     return 0.05
    # elif round_idx <= 110:
    #     return 0.01
    # else:
    #     return 0.005


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, float]:
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

def aggregate(client_states, client_sizes):
    """
    Standard FedAvg aggregation (Full-Precision).

    Args:
        client_states: List[Dict[str, Tensor]]
            Each element is a client's FP state_dict (shadow weights).
        client_sizes: List[int]
            Number of local samples on each client.

    Returns:
        global_state: Dict[str, Tensor]
            Aggregated FP global model parameters.
    """

    assert len(client_states) > 0
    assert len(client_states) == len(client_sizes)

    total_samples = sum(client_sizes)
    weights = [n / total_samples for n in client_sizes]

    # initialize with zeros
    global_state = {}

    for key in client_states[0].keys():
    #     if ("running_mean" in key) or ("running_var" in key) or ("num_batches_tracked" in key):
    #         # 选一种策略即可：
    #         # 1) 保持上一轮全局的（需要你外部传入 prev_global_state）
    #         # 2) 或者直接用第一个 client 的（更粗暴但能稳定验证）
    #         global_state[key] = client_states[0][key].clone()
    #         continue

        acc = None
        
        for state, w in zip(client_states, weights):
            tensor = state[key].to(dtype=torch.float32)
            #print("std of uploaded tensor:", tensor.std().item())
            
            if acc is None:
                acc = w * tensor
            else:
                acc += w * tensor
        #print("hahahah ",key,acc)
        global_state[key] = acc

    return global_state


def binary_aggregate(client_hat_states, nks, bin_param_names, fp_param_names):
    """
    client_hat_states: 每个 client 上传的 state_dict（binary部分已sign，fp部分为fp）
    返回：global_hat（binary部分仍为±1；fp部分为fp）
    """
    weights = [nk / max(sum(nks), 1) for nk in nks]
    bin_set = set(bin_param_names)
    fp_set = set(fp_param_names)

    template = client_hat_states[0]
    global_hat = {}

    for k in template.keys():
        if k in bin_set:
            # binary: weighted sum -> sign
            acc = None
            for sd, w in zip(client_hat_states, weights):
                t = sd[k].to(dtype=torch.float32)
                acc = t * w if acc is None else acc + t * w
            global_hat[k] = sign_tensor(acc)

        elif k in fp_set:
            # fp trainable: normal FedAvg
            acc = None
            for sd, w in zip(client_hat_states, weights):
                t = sd[k].to(dtype=torch.float32)
                acc = t * w if acc is None else acc + t * w
            global_hat[k] = acc

        else:
            # buffers (e.g., bn.running_mean/var/num_batches_tracked): copy is simplest & stable
            global_hat[k] = template[k].detach().clone()

    return global_hat

def cab_aggregate(client_hat_states, nks, bin_param_names, fp_param_names):
    """
    返回 (new_hat, tilde)
    - tilde: sign 前的加权和（binary weights 的共识强度，值域[-1,1]）
    - new_hat: sign(tilde)
    """
    total = max(sum(nks), 1)
    weights = [nk / total for nk in nks]
    bin_set = set(bin_param_names)
    fp_set = set(fp_param_names)

    template = client_hat_states[0]
    tilde = {}
    new_hat = {}

    for k in template.keys():
        if k in bin_set:
            acc = None
            for sd, w in zip(client_hat_states, weights):
                t = sd[k].detach().float()  # 这里 sd[k] 已经是 ±1
                acc = t * w if acc is None else acc + t * w
            tilde[k] = acc                  # ✅ 保留 sign 前的 acc（通常不是 ±1）
            new_hat[k] = sign_tensor(acc)   # ✅ new_hat 才 sign
        elif k in fp_set:
            acc = None
            for sd, w in zip(client_hat_states, weights):
                t = sd[k].detach().float()
                acc = t * w if acc is None else acc + t * w
            tilde[k] = acc
            new_hat[k] = acc
        else:
            tilde[k] = template[k].detach().clone()
            new_hat[k] = template[k].detach().clone()

    return new_hat, tilde

def aggregate_scales(client_scale_list, nks, bin_param_names, scale_min=1e-6, scale_max=1e6):
    weights = [nk / max(sum(nks), 1) for nk in nks]
    s = {}
    for name in bin_param_names:
        val = 0.0
        for sk, w in zip(client_scale_list, weights):
            val += float(sk[name]) * float(w)
        # clamp for safety
        val = float(max(scale_min, min(scale_max, val)))
        s[name] = val
    return s



def aggregate_bin_sign_and_fp(
    client_hat_states,  # 每个客户端：binary param 是 ±1（或 float），fp param 是 float
    nks,
    bin_param_names,
    fp_param_names,
):
    """
    返回:
      tilde_bin[name] = weighted avg of client binary signs (float in [-1,1])
      agg_fp[name]    = FedAvg on FP params
    """
    weights = [nk / max(sum(nks), 1) for nk in nks]

    # template
    template = client_hat_states[0]
    agg_fp = {k: v.clone() for k, v in template.items()}
    tilde_bin = {}

    # binary: accumulate float average (before sign)
    for name in bin_param_names:
        acc = None
        for st, w in zip(client_hat_states, weights):
            t = st[name].to(dtype=torch.float32)
            acc = t * w if acc is None else acc + t * w
        tilde_bin[name] = acc  # float tensor

    # fp: FedAvg
    for name in fp_param_names:
        acc = None
        for st, w in zip(client_hat_states, weights):
            t = st[name].to(dtype=torch.float32)
            acc = t * w if acc is None else acc + t * w
        agg_fp[name] = acc

    return tilde_bin, agg_fp
