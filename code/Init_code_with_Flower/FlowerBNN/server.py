#This is the server of FL with initial Flower
import flwr as fl
from flwr.server.strategy import FedAvg
from flwr.server.client_proxy import ClientProxy
from flwr.common.parameter import ndarrays_to_parameters, parameters_to_ndarrays
from flwr.common import (
    Code,
    DisconnectRes,
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    Parameters,
    ReconnectIns,
    Scalar,
)
from typing import List,Tuple,Union,Dict,Optional
import numpy as np


def aggregate(weights,num):
    #print("AGG",len(weights),len(num))
    ret = [weight*num[0] for weight in weights[0]]
    for i in range(1,len(weights)):
        for j in range(len(ret)):
            ret[j] += weights[i][j]*num[i]
    Sum = sum(num)
    for idx,w in enumerate(ret):
        if w.ndim > 1:
            ret[idx] = np.sign(w).astype(int)
        else:
            ret[idx] = w/Sum
    return ret


class CustomedFedAvg(FedAvg):
    def __init__(
        self,
        min_available_clients=3,
        min_fit_clients=3,
        ):
        super().__init__()
        self.min_available_clients = min_available_clients
        self.min_fit_clients = min_fit_clients
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) :
        #print("IN Aggregate_fit Server_round:",server_round)
        weights_results = []
        data_num = []
        for _, fit_res in results:
            weights_results.append(parameters_to_ndarrays(fit_res.parameters))
            data_num.append(fit_res.num_examples)
        
        parameters_aggregated = ndarrays_to_parameters(aggregate(weights_results,data_num))

        return parameters_aggregated,{}
        
if __name__ == '__main__':
    client_num = 3
    num_round = 60
    strategy = CustomedFedAvg(client_num,client_num)
    fl.server.start_server(
    server_address="0.0.0.0:8080",
    strategy=strategy,
    config=fl.server.ServerConfig(num_rounds=num_round)
    )