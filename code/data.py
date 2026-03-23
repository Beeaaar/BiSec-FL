import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import Subset
from typing import Dict, List, Tuple, Optional
import json
import torch
import os
from torch.utils.data import Dataset
import glob

class FEMNISTDataset(Dataset):
    def __init__(self, json_files, transform=None):
        if isinstance(json_files, str):
            json_files = [json_files]

        self.transform = transform
        self.images = []
        self.targets = []
        self.writers = []

        for jf in json_files:
            with open(jf, "r") as f:
                data = json.load(f)

            for user in data["users"]:
                xs = data["user_data"][user]["x"]
                ys = data["user_data"][user]["y"]

                for x, y in zip(xs, ys):
                    img = np.array(x, dtype=np.float32).reshape(28, 28)
                    self.images.append(img)
                    self.targets.append(int(y))
                    self.writers.append(user)

        # 不要在这里就转成 CUDA tensor，留给 DataLoader
        # list[np.ndarray] -> np.ndarray
        self.images = np.stack(self.images, axis=0).astype(np.float32)
        self.targets = np.array(self.targets, dtype=np.int64)

        # np.ndarray -> torch.Tensor
        self.images = torch.from_numpy(self.images)
        self.targets = torch.from_numpy(self.targets)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        """
        必须实现！
        """
        x = self.images[idx]          # [28, 28]
        y = self.targets[idx]

        if self.transform is not None:
            # ToTensor 期望 H×W numpy / PIL
            x = self.transform(x.numpy())

        return x, y

def get_datasets(name: str, data_root: str,shards = [0]):
    name = name.lower()
    if name == "mnist":
        tfm = transforms.Compose([transforms.ToTensor()])
        train = datasets.MNIST(root=data_root, train=True, download=True, transform=tfm)
        test = datasets.MNIST(root=data_root, train=False, download=True, transform=tfm)
        return train, test, 10

    if name == "emnist":
        # 这里用 EMNIST(byclass) 做 centralized stand-in / 伪 federated 切分
        tfm = transforms.Compose([transforms.ToTensor()])
        train = datasets.EMNIST(root=data_root, split="byclass", train=True, download=True, transform=tfm)
        test = datasets.EMNIST(root=data_root, split="byclass", train=False, download=True, transform=tfm)
        return train, test, 62

    if name == "femnist":
        tfm = transforms.Compose([transforms.ToTensor()])
        print("TO MANY FILES?")
        train_jsons = [
            os.path.join(
                data_root,
                f"leaf/data/femnist/data/train/all_data_{i}_niid_0_keep_5_train_9.json"
            )
            for i in shards
        ]

        test_jsons = [
            os.path.join(
                data_root,
                f"leaf/data/femnist/data/test/all_data_{i}_niid_0_keep_5_test_9.json"
            )
            for i in shards
        ]


        assert len(train_jsons) > 0
        assert len(test_jsons) > 0

        train = FEMNISTDataset(train_jsons, transform=tfm)
        test = FEMNISTDataset(test_jsons, transform=tfm)
        print("Got FEMNIST DATASET")
        return train, test, 62

    if name == "cifar10":
        tfm_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
        tfm_test = transforms.Compose([transforms.ToTensor()])
        # 注意：如果你服务器不能联网，就把 download=True 改成 False，并手动放好 cifar-10-batches-py
        train = datasets.CIFAR10(root=data_root, train=True, download=True, transform=tfm_train)
        test = datasets.CIFAR10(root=data_root, train=False, download=True, transform=tfm_test)
        return train, test, 10

    raise ValueError(f"Unknown dataset: {name}")

def partition_iid(train_set, num_clients: int, seed: int) -> List[List[int]]:
    print("DATA NUM :",len(train_set))
    rng = np.random.default_rng(seed)
    idxs = np.arange(len(train_set))
    rng.shuffle(idxs)
    splits = np.array_split(idxs, num_clients)
    return [s.tolist() for s in splits]


def partition_dirichlet(train_set, num_clients: int, alpha: float, seed: int) -> List[List[int]]:
    # label-based Dirichlet partition
    print(f"label-based dirichlet partition, numclients:{num_clients},alpha:{alpha}")
    rng = np.random.default_rng(seed)
    targets = np.array(train_set.targets if hasattr(train_set, "targets") else train_set.labels)

    num_classes = int(targets.max()) + 1
    idx_by_class = [np.where(targets == c)[0] for c in range(num_classes)]
    for c in range(num_classes):
        rng.shuffle(idx_by_class[c])

    client_indices = [[] for _ in range(num_clients)]
    for c in range(num_classes):
        idx_c = idx_by_class[c]
        if len(idx_c) == 0:
            continue
        proportions = rng.dirichlet(alpha * np.ones(num_clients))
        proportions = (np.cumsum(proportions) * len(idx_c)).astype(int)[:-1]
        splits = np.split(idx_c, proportions)
        for k in range(num_clients):
            client_indices[k].extend(splits[k].tolist())

    for k in range(num_clients):
        rng.shuffle(client_indices[k])
    return client_indices


def partition_femnist(
    train_set,
    num_clients: int,
    seed: int = 0
):
    """
    FEMNIST partition with writer merging.
    Each client consists of multiple writers.
    """
    print(f"FEMNIST partition | merged writers → {num_clients} clients")

    rng = np.random.default_rng(seed)

    if not hasattr(train_set, "writers"):
        raise ValueError("FEMNISTDataset must have `writers` attribute")

    writers = np.array(train_set.writers)
    unique_writers = np.unique(writers)

    rng.shuffle(unique_writers)

    # 将 writer 均匀分配给 clients
    writer_splits = np.array_split(unique_writers, num_clients)

    client_indices = []

    for ws in writer_splits:
        idxs = []
        for w in ws:
            idxs.extend(np.where(writers == w)[0].tolist())
        rng.shuffle(idxs)
        client_indices.append(idxs)

    return client_indices


def partition_femnist_balanced(train_set, num_clients, seed=0):
    rng = np.random.default_rng(seed)

    writers = np.array(train_set.writers)
    unique_writers, counts = np.unique(writers, return_counts=True)

    # 打乱 writer 顺序
    perm = rng.permutation(len(unique_writers))
    unique_writers = unique_writers[perm]
    counts = counts[perm]

    client_bins = [[] for _ in range(num_clients)]
    client_sizes = [0] * num_clients

    for w, c in zip(unique_writers, counts):
        k = np.argmin(client_sizes)
        client_bins[k].append(w)
        client_sizes[k] += c

    client_indices = []
    for ws in client_bins:
        idxs = []
        for w in ws:
            idxs.extend(np.where(writers == w)[0])
        rng.shuffle(idxs)
        client_indices.append(idxs)
    print("SPLIT FEMNIST NIID BALANCED")
    return client_indices
