from dataclasses import dataclass

@dataclass
class FLConfig11:
    dataset: str = "mnist"
    data_root:str = "./data"
    model: str = "MLP"
    num_clients: int = 20
    clients_per_round: int = 5
    rounds: int = 100
    local_epochs: int = 3
    batch_size: int = 64
    lr: float = 0.05
    device: str = "cuda"
    num_workers:int = 4
    partition:str = "iid"
    seed: int = 0
    out_root:str = "./runs_fl"
    method:str = "fp_fedavg"


@dataclass
class FLConfig12:
    dataset: str = "emnist"
    data_root: str = "./data"
    model: str = "LeNetBN"
    num_clients: int = 20
    clients_per_round: int = 10
    rounds: int = 100
    local_epochs: int = 1
    batch_size: int = 64
    lr: float = 0.03
    device: str = "cuda"
    num_workers: int = 4
    partition: str = "dirichlet"
    seed: int = 0
    out_root: str = "./runs_fl"
    method: str = "fp_fedavg"
    alpha:float = 0.5

@dataclass
class FLConfig13:
    dataset: str = "cifar10"
    data_root: str = "./data"
    model: str = "ResNet18"
    num_clients: int = 20
    clients_per_round: int = 5
    rounds: int = 250
    local_epochs: int = 3
    batch_size: int = 64
    lr: float = 0.05
    device: str = "cuda"
    num_workers: int = 4
    partition: str = "iid"
    seed: int = 0
    out_root: str = "./runs_fl"
    method: str = "fp_fedavg"


@dataclass
class FLConfig_BinLocal11:
    dataset: str = "mnist"
    data_root: str = "./data"
    model: str = "BinaryMLP"
    num_clients: int = 20
    clients_per_round: int = 5
    rounds: int = 150
    local_epochs: int = 3
    batch_size: int = 64
    lr: float = 0.05
    device: str = "cuda"
    num_workers: int = 4
    partition: str = "iid"
    seed: int = 0
    out_root: str = "./runs_fl"
    method: str = "bin_local_fedavg"

@dataclass
class FLConfig_BinLocal12:
    dataset: str = "emnist"
    data_root: str = "./data"
    model: str = "BinaryLeNetBN"
    num_clients: int = 20
    clients_per_round: int = 5
    rounds: int = 100
    local_epochs: int = 1
    batch_size: int = 64
    lr: float = 0.03
    device: str = "cuda"
    num_workers: int = 4
    partition: str = "dirichlet"
    seed: int = 0
    out_root: str = "./runs_fl"
    method: str = "bin_local_fedavg"
    alpha:float = 0.1
@dataclass
class FLConfig_BinLocal13:
    dataset: str = "cifar10"
    data_root: str = "./data"
    model: str = "BinaryResNet18"
    num_clients: int = 20
    clients_per_round: int = 5
    rounds: int = 250
    local_epochs: int = 3
    batch_size: int = 64
    lr: float = 0.05
    device: str = "cuda"
    num_workers: int = 4
    partition: str = "iid"
    seed: int = 0
    out_root: str = "./runs_fl"
    method: str = "bin_local_fedavg"


@dataclass
class FLConfig_BinFL31:
    dataset: str = "mnist"
    data_root: str = "./data"
    model: str = "BinaryMLP"
    num_clients: int = 20
    clients_per_round: int = 5
    rounds: int = 100
    local_epochs: int = 3
    batch_size: int = 64
    lr: float = 0.05
    device: str = "cuda"
    num_workers: int = 4
    partition: str = "iid"
    seed: int = 0
    out_root: str = "./runs_fl"
    method: str = "bin_fedavg"

@dataclass
class FLConfig_BinFL32:
    dataset: str = "emnist"
    data_root: str = "./data"
    model: str = "BinaryLeNetBN"
    num_clients: int = 20
    clients_per_round: int = 5
    rounds: int = 100
    local_epochs: int = 1
    batch_size: int = 64
    lr: float = 0.03
    device: str = "cuda"
    num_workers: int = 4
    partition: str = "dirichlet"
    seed: int = 0
    out_root: str = "./runs_fl"
    method: str = "bin_fedavg"
    alpha:float = 0.1

@dataclass
class FLConfig_BinFL33:
    dataset: str = "cifar10"
    data_root: str = "./data"
    model: str = "BinaryResNet18"
    num_clients: int = 20
    clients_per_round: int = 5
    rounds: int = 250
    local_epochs: int = 3
    batch_size: int = 64
    lr: float = 0.05
    device: str = "cuda"
    num_workers: int = 4
    partition: str = "iid"
    seed: int = 0
    out_root: str = "./runs_fl"
    method: str = "bin_fedavg"


@dataclass
class FLConfig_CabFL41:
    dataset: str = "mnist"
    data_root: str = "./data"
    model: str = "BinaryMLP"
    num_clients: int = 20
    clients_per_round: int = 5
    rounds: int = 100
    local_epochs: int = 3
    batch_size: int = 64
    lr: float = 0.001
    device: str = "cuda"
    num_workers: int = 4
    partition: str = "iid"
    seed: int = 0
    out_root: str = "./runs_fl"
    method: str = "cab_fl"

@dataclass
class FLConfig_CabFL42:
    dataset: str = "emnist"
    data_root: str = "./data"
    model: str = "BinaryLeNetBN"
    num_clients: int = 20
    clients_per_round: int = 15
    rounds: int = 100
    local_epochs: int = 1
    batch_size: int = 64
    lr: float = 0.03
    device: str = "cuda"
    num_workers: int = 4
    partition: str = "dirichlet"
    seed: int = 0
    out_root: str = "./runs_fl"
    method: str = "cab_fl"
    alpha:float = 0.1

@dataclass
class FLConfig_CabFL43:
    dataset: str = "cifar10"
    data_root: str = "./data"
    model: str = "BinaryResNet18"
    num_clients: int = 20
    clients_per_round: int = 5
    rounds: int = 250
    local_epochs: int = 3
    batch_size: int = 64
    lr: float = 0.1
    device: str = "cuda"
    num_workers: int = 4
    partition: str = "iid"
    seed: int = 0
    out_root: str = "./runs_fl"
    method: str = "cab_fl"



@dataclass
class testConfig:
    dataset: str = "femnist"
    data_root: str = "./data"
    model: str = "BinaryMLP"
    num_clients: int = 20
    clients_per_round: int = 5
    rounds: int = 100
    local_epochs: int = 3
    batch_size: int = 128
    lr: float = 0.05
    device: str = "cuda"
    num_workers: int = 4
    partition: str = "iid"
    seed: int = 0
    out_root: str = "./runs_fl"
    method: str = "bin_fedavg"
    alpha:float = 0.1



@dataclass
class FLConfig_CabFL_FEMNIST_20_5:
    dataset: str = "femnist"
    data_root: str = "./data"
    model: str = "BinaryLeNetBN"

    # federated setting
    num_clients: int = 20
    clients_per_round: int = 5
    rounds: int = 100

    # optimization (stability-first for FEMNIST)
    local_epochs: int = 1
    batch_size: int = 64
    lr: float = 0.02

    # system
    device: str = "cuda"
    num_workers: int = 4

    # partition
    partition: str = "femnist"   # 关键：真实 user-level non-IID
    seed: int = 0
    shards = [0,1,2,3,4,5,6]

    # method
    out_root: str = "./runs_fl"
    method: str = "cab_fl"



@dataclass
class FLConfig_CabFL_FEMNIST_100_10:
    dataset: str = "femnist"
    data_root: str = "./data"
    model: str = "BinaryLeNetBN"

    # federated setting
    num_clients: int = 100
    clients_per_round: int = 10
    rounds: int = 100

    # optimization (slightly more aggressive)
    local_epochs: int = 1
    batch_size: int = 64
    lr: float = 0.2

    # system
    device: str = "cuda"
    num_workers: int = 4

    # partition
    partition: str = "femnist"
    seed: int = 0
    shards = [0,1,2,3,4,5,6,7,8,9,10,11,12,13]
    # method
    out_root: str = "./runs_fl"
    method: str = "cab_fl"

    # FEMNIST 不使用 Dirichlet
    alpha: float = 0.0


@dataclass
class FLConfig_BinFL_FEMNIST_20_5:
    dataset: str = "femnist"
    data_root: str = "./data"
    model: str = "BinaryLeNetBN"

    # federated setting
    num_clients: int = 20
    clients_per_round: int = 5
    rounds: int = 150

    # optimization (Bin-FedAvg is more fragile on FEMNIST)
    local_epochs: int = 1
    batch_size: int = 64
    lr: float = 0.02

    # system
    device: str = "cuda"
    num_workers: int = 4

    # partition
    partition: str = "femnist"
    seed: int = 0
    shards = [0,1,2,3,4,5,6]
    # method
    out_root: str = "./runs_fl"
    method: str = "bin_fedavg"

@dataclass
class FLConfig_BinFL_FEMNIST_100_10:
    dataset: str = "femnist"
    data_root: str = "./data"
    model: str = "BinaryLeNetBN"

    # federated setting
    num_clients: int = 100
    clients_per_round: int = 10
    rounds: int = 100

    # optimization
    local_epochs: int = 1
    batch_size: int = 64
    lr: float = 0.1

    # system
    device: str = "cuda"
    num_workers: int = 4

    # partition
    partition: str = "femnist"
    shards = [0,1,2,3,4,5,6,7,8,9,10,11,12,13]
    seed: int = 0

    # method
    out_root: str = "./runs_fl"
    method: str = "bin_fedavg"

@dataclass
class FLConfig_BinLocal_FEMNIST_20_5:
    dataset: str = "femnist"
    data_root: str = "./data"
    model: str = "BinaryLeNetBN"

    # federated setting
    num_clients: int = 20
    clients_per_round: int = 5
    rounds: int = 100

    # optimization (Bin-Local is still sensitive on FEMNIST)
    local_epochs: int = 1
    batch_size: int = 64
    lr: float = 0.03

    # system
    device: str = "cuda"
    num_workers: int = 4

    # partition
    partition: str = "femnist"
    shards = [0,1,2,3,4,5]
    #shards = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30]
    seed: int = 0

    # method
    out_root: str = "./runs_fl"
    method: str = "bin_local_fedavg"

@dataclass
class FLConfig_BinLocal_FEMNIST_100_10:
    dataset: str = "femnist"
    data_root: str = "./data"
    model: str = "BinaryLeNetBN"

    # federated setting
    num_clients: int = 100
    clients_per_round: int = 10
    rounds: int = 100

    # optimization
    local_epochs: int = 1
    batch_size: int = 64
    lr: float = 0.03

    # system
    device: str = "cuda"
    num_workers: int = 4

    # partition
    partition: str = "femnist"
    seed: int = 0
    shards = [10,18]
    # method
    out_root: str = "./runs_fl"
    method: str = "bin_local_fedavg"

@dataclass
class FLConfig_FEMNIST_20_5:
    dataset: str = "femnist"
    data_root: str = "./data"
    model: str = "LeNetBN"

    # federated setting
    num_clients: int = 20
    clients_per_round: int = 5
    rounds: int = 100

    # optimization (FP-FedAvg on FEMNIST)
    local_epochs: int = 1
    batch_size: int = 64
    lr: float = 0.01

    # system
    device: str = "cuda"
    num_workers: int = 4

    # partition
    partition: str = "femnist"
    seed: int = 0
    shards = [0,1,2,3,4,5,6]
    # method
    out_root: str = "./runs_fl"
    method: str = "fp_fedavg"

@dataclass
class FLConfig_FEMNIST_100_10:
    dataset: str = "femnist"
    data_root: str = "./data"
    model: str = "LeNetBN"

    # federated setting
    num_clients: int = 100
    clients_per_round: int = 10
    rounds: int = 100

    # optimization
    local_epochs: int = 1
    batch_size: int = 64
    lr: float = 0.08

    # system
    device: str = "cuda"
    num_workers: int = 4

    # partition
    partition: str = "femnist"
    seed: int = 0
    shards = [0,1,2,3,4,5,6,7,8,9,10,11,12,13]
    # method
    out_root: str = "./runs_fl"
    method: str = "fp_fedavg"