from config import FLConfig_BinLocal11,FLConfig_BinLocal12,FLConfig_BinLocal13,FLConfig_BinLocal_FEMNIST_20_5,FLConfig_BinLocal_FEMNIST_100_10
from util import build_model,set_seed,evaluate,train,aggregate,get_device,get_global_lr
from data import get_datasets,partition_iid,partition_dirichlet,partition_femnist_balanced

import csv
import numpy as np
from torch.utils.data import DataLoader, Subset
import os 
import sys


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

    metrics_path = os.path.join(run_dir, str(cfg.lr)+'_'+str(cfg.local_epochs)+'_'+str(cfg.num_clients)+'_'+str(cfg.clients_per_round)+"_metrics.csv")
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

        # ===== 2. Client 本地训练 =====
        for k in selected_clients:
            local_model = build_model(cfg.model, num_classes).to(device)
            local_model.load_state_dict(global_state)

            subset = Subset(train_set, client_indices[k])

            #cfg.lr = get_global_lr(r)
            train_acc,local_state = train(
                model=local_model,
                train_subset=subset,
                device = device,
                cfg=cfg
            )

            client_states.append(local_state)
            client_sizes.append(len(client_indices[k]))
            client_train_accs.append(train_acc)

        mean_train_acc = float(np.mean(client_train_accs)) if client_train_accs else 0.0

        # ===== 3. Server 端 FedAvg =====
        global_state = aggregate(client_states, client_sizes)
        global_model.load_state_dict(global_state)

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
    
    #cfg = FLConfig_BinLocal_FEMNIST_100_10()
    cfg = FLConfig_BinLocal12()
    #cfg = FLConfig_BinLocal_FEMNIST_20_5()
    set_seed(cfg.seed)

    device = get_device(cfg.device)
    if cfg.partition == 'femnist':
        train_set, test_set, num_classes = get_datasets(cfg.dataset, cfg.data_root,cfg.shards)
    else:
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

    # print("TOT TRAIN DATA ",len(train_set))
    # for num in client_idxs:
    #     print("DATA NUM",len(num))
    history = fl_round_loop(cfg,train_set,test_loader,client_idxs,num_classes,device)    


if __name__ == '__main__':
    main()