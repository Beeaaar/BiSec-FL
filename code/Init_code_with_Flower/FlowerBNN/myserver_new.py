import os
import flwr as fl
from rlwe import RLWE,Rq
from flwr.server.strategy import FedAvg
from flwr.server.client_proxy import ClientProxy
from flwr.common.parameter import ndarrays_to_parameters, parameters_to_ndarrays
import numpy as np
import utils
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
def aggregate(weights,data_num,Max=100)-> List:
        Min_num = min(data_num)
        for i in range(len(data_num)):
            data_num[i] = min(Max,data_num[i]//Min_num)
        ret = weights[0]*data_num[0]
        #print("!!!!!!!!!!!!TYPE:",type(weights[0]),type(ret),type(ret[0]))
        for i in range(1,len(data_num)):
            ret += weights[i]*data_num[i]    
        Sum = sum(data_num)
        ret = [np.sign(weight/Sum).astype(int) for weight in ret]
        #ret = [weight/Sum for weight in ret]
        flat_weight = []
        for weight in ret:
            flat_weight.extend(weight.flatten())
        #print(type(ret))
            #metrics_aggregated = {}
        return flat_weight

def aggregate_new(weights,data_num,Max=100)-> List:
        print("WHY IDX WRONG ",type(weights),len(weights))
        idx = weights[0][-1]
        idx = 5
        ####记得改！！！要传过来！！！
        print("In AGG,",idx)
        print("type,",type(weights[0]),type(weights[0][5:]))
        meaning = [weights[i][idx:-1] for i in range(len(weights))]
        pure_weights = [weights[i][:idx] for i in range(len(weights))]
        Min_num = min(data_num)
        for i in range(len(data_num)):
            data_num[i] = min(Max,data_num[i]//Min_num)
        retw = pure_weights[0]*data_num[0]
        retm = meaning[0]*data_num[0]
        #print("!!!!!!!!!!!!TYPE:",type(weights[0]),type(ret),type(ret[0]))
        for i in range(1,len(data_num)):
            retw = [w + pw*data_num[i] for w,pw in zip(retw,pure_weights[i])]
            retm = [m + pm*data_num[i] for m,pm in zip(retm,meaning[i])]
            Sum = sum(data_num)
        for m in retm:
            print("到底是什么啊",type(m),m.shape)
        retw = [np.sign(weight/Sum).astype(int) for weight in retw]
        retm = [(m/Sum).astype(int) for m in retm]
        #ret = [weight/Sum for weight in ret]
        flat_ret = []
        for weight in retw:
            flat_ret.extend(weight.flatten())
        for m in retm:
           flat_ret.extend(m.flatten())  
        #print(type(ret))
            #metrics_aggregated = {}
        return flat_ret
class CustomFedAvg(FedAvg):
    def __init__(self, rlwe_instance: RLWE, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rlwe = rlwe_instance
    
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) :
        print("IN Aggregate_fit Server_round:",server_round)
        weights_results = []
        data_num = []
        for _, fit_res in results:
            weights_results.append(parameters_to_ndarrays(fit_res.parameters))
            data_num.append(fit_res.num_examples)
        print("DATA NUM:",data_num)    
        parameters_aggregated = aggregate_new(weights_results,data_num)

        #这里没有处理到二值化
        #print("aggragate result:",parameters_aggregated)
        return parameters_aggregated


def fit_round(server_round: int):
    """Send round number to client."""
    return {"server_round": server_round}

def Init_RLWE(Scale,num_clients):
    #参数问题还没解决，先用着,也暂时就确定了模型参数
    #para_num = 9692746,2**23=8388608
    n = 2**13
    #q = 100_000_000_003 
    #t = 200_000_001
    t = utils.next_prime(num_clients*(10**Scale)*5)
    q = utils.next_prime(t*20)
    std = 3
    rlwe = RLWE(n,q,t,std)
    print("IN RLWE_INIT ",n,q,t)
    return rlwe
    std = 3
    #密钥s的生成问题也和原文不太一致，待解决
    return RLWE(n,q,t,std)


def Customed_strategy(client_num):
    Scale = 7
    rlwe = Init_RLWE(Scale,client_num)
    #这个自定义策略涉及聚合方式，后面要改
    strategy = CustomFedAvg(
        min_available_clients=client_num,
        min_fit_clients=client_num,
        #服务端没数据，不设置evaluate_fn
        on_fit_config_fn=fit_round,
        rlwe_instance=rlwe,
    )
    #client_manager没改动，注释掉应该没影响
    #client_manager = fl.server.SimpleClientManager()
    return strategy
    
if __name__ =='__main__':
    client_num = 3
    strategy = Customed_strategy(client_num)
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        #server = server,#需要自定义的server类
        strategy=strategy,
        config=fl.server.ServerConfig(num_rounds=3),
    )