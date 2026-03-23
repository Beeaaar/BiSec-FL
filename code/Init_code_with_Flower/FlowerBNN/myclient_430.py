from flwr.common import NDArrays
import argparse
import numpy as np
import flwr as fl
import torch
from collections import OrderedDict
from load_MNIST import load_data,load_part_data
from Net import *
from rlwe import RLWE,Rq
import utils
from binarized_modules import binarized
from typing import List, Tuple,Dict
import Split_MNIST
parser = argparse.ArgumentParser(description='Myclient Datapath')

parser.add_argument('--datapth',metavar='DATA_PATH',
                    default='./data/MNIST/client1',
                    help = 'data_path')

args = parser.parse_args()
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
trainloader,testloader = load_data()
#trainloader,testloader = load_part_data(args.datapth)
#trainloader,testloader = Split_MNIST.Load_part_data(idx = 2)
net = Binary_CF_superMini().to(DEVICE)
def Init_RLWE(Scale,num_clients):
    #参数问题还没解决，先用着,也暂时就确定了模型参数
    #para_num = 9692746,2**23=8388608
    n = 2**13
    #q = 100_000_000_003 
    #t = 200_000_001
    #std = 3
    t = utils.next_prime(num_clients*(10**Scale)*3)
    q = utils.next_prime(t*20)
    std = 3
    rlwe = RLWE(n,q,t,std)
    print("IN RLWE_INIT ",n,q,t)
    return rlwe
    #密钥s的生成问题也和原文不太一致，待解决
    #return RLWE(n,q,t,std)

class MNISTClient(fl.client.NumPyClient):
    def __init__(self, rlwe_instance: RLWE, WEIGHT_DECIMALS: int, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.rlwe = rlwe_instance
                self.allpub = None
                self.model_shape = None
                self.weight_shape = None
                self.model_length = None
                self.flat_params = None
                self.WEIGHT_DECIMALS = WEIGHT_DECIMALS
                self.meaning_num = None
                #如果要加权这里是不是要多一个数据量参数或者权重。
                self.model = net
                #utils.set_initial_params(self.model)这里包括了Loss和optimizer信息，我们用pytorch直接在类里加上吧
    
    #-> List[ndarray[Any, dtype[Any]]]    
    #get和set都要缩放参数，源码重构了keras的get_weights和set_weights    
    def get_parameters(self, config) :
        print("IN testclient get_parameters")
        return get_weights(self.model)
        weights =[val.cpu().numpy() for _, val in self.model.state_dict().items()]
        for w in weights:
            print(w.shape)
        return weights
    #这里先按照原来的可行写法搞，后面看看能不能简化
    def set_parameters(self,bias,weights,meaning):
        #set_weight(self.model,parameters)
        set_weight_and_bias_and_meaning(self.model,bias,weights,meaning)
        '''params_dict = zip(net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)'''
    #这个设置很神奇，应该是在fit之前就专门传过参数把网络设置好了，所以这里没给参数，直接网络去训练就行了
    def get_binary_weight_meaning(self):
        weights = []
        weight_shape = []
        meaning = []
        meaning_shape = []
        L = 0
        for name, val in self.model.state_dict().items():
            if 'num_batches_tracked' in name: continue
            if 'bias' not in name:
                if 'weight' in name:
                    weights.append(binarized(val).cpu().numpy())
                    weight_shape.append(weights[-1].shape)
                else:
                    meaning.append(val.cpu().numpy()*10000)
                    meaning_shape.append(meaning[-1].shape)
                    L += len(meaning[-1])
            '''elif 'bias' in name:
                bias.append(val.cpu())'''
        self.weight_shape = weight_shape + meaning_shape
        weights += meaning
        print("number of meaning:",L)
        self.meaning_num = L
        weights.append(L)
        return weights  
    def fit(self,parameters,config):
        #print("\n\nFit WHAT?\n\n",print(parameters))
        
        #self.set_parameters(parameters)
        train(self.model,trainloader,DEVICE,epochs=5)
        Loss,Acc = test(self.model,testloader,DEVICE)
        print('Test after fit in clients: Average loss: {:.4f}, Accuracy: {:.2f}% '.format(Loss,Acc))
        #这里不用传参数回去了,因为是常规流程调用的，如果传了就相当于没加密的传回去了，考虑在这里展开加密过程，传回去之后需要重写本来的聚合过程。通过serde，message_handler等函数
        
        return self.get_binary_weight_meaning(), len(trainloader.dataset), {}
    
    #好像没用到这个evaluate函数
    def evaluate(self, parameters, config):    
        self.set_parameters(parameters)
        loss, accuracy = test(self.model, testloader,DEVICE) 
        print("IN myclient:76 ??????When to evaluate????????",loss,accuracy)
        return float(loss), len(testloader.dataset), {"accuracy": float(accuracy)}

    #下面是跟加密有关系的东西了
    #vector为什么在服务器那边生成了多项式转成list到这还要转回多项式，通信的时候不能传多项式吗

    def example_response(self, question: str, l: List[int]) -> Tuple[str, int]:
            response = "In example_responce,about encryption"
            answer = sum(l)
            return response, answer
    def generate_pubkey(self,vector_a):
        print(f"In myclient:82,vector_a: {vector_a[:8]}")
        vector_a = self.rlwe.list_to_poly(vector_a,"q")
        self.rlwe.set_vector_a(vector_a)
        (_, pub) = rlwe.generate_keys()
        
        
        #发回回去的是b，是公钥的一部分
        return pub[0].poly_to_list()
         
    def store_aggregated_pubkey(self, allpub) -> bool:
            aggregated_pubkey = self.rlwe.list_to_poly(allpub, "q")
            self.allpub = (aggregated_pubkey, self.rlwe.get_vector_a())
            print(f"client allpub: {self.allpub[:8]}")
            return True
    
    def encrypt_parameters(self,request):
        print(f"request msg is: {request}")
        #这里如果分批处理了，需要改
        flattened_weights, self.model_shape = utils.get_flat_bias(self.model)
        flattened_weights, self.model_length = utils.pad_to_power_of_2(flattened_weights, self.rlwe.n, self.WEIGHT_DECIMALS)
        #print(f"Client old plaintext: {self.flat_params[925:935]}") if self.flat_params is not None else None
        #print(f"Client new plaintext: {flattened_weights[925:935]}")
        poly_weights = Rq(np.array(flattened_weights), self.rlwe.t)
        if request == "gradient":
            gradient = list(np.array(flattened_weights) - np.array(self.flat_params))
            #print(f"Client gradient: {gradient[925:935]}")
            poly_weights = Rq(np.array(gradient), self.rlwe.t)
        c0, c1 = self.rlwe.encrypt(poly_weights, self.allpub)
        c0 = list(c0.poly.coeffs)
        c1 = list(c1.poly.coeffs)
        print("In myclient:139,plain_text(para or gra):",poly_weights.poly_to_list()[:8])
        print(f"c0: {c0[:8]}")
        print(f"c1: {c1[:8]}")
        return c0, c1

    def compute_decryption_share(self,csum1):
        std = 5
        print("In myclient:114,Csum1 from server:",csum1[:8])
        csum1_poly = self.rlwe.list_to_poly(csum1,"q")
        err = Rq(np.round(std*np.random.randn(self.rlwe.n)),self.rlwe.p)
        d1 = self.rlwe.decrypt(csum1_poly,self.rlwe.s,err)
        d1 = list(d1.poly.coeffs)
        #print("In myclient:130:decryp err:",err.poly_to_list()[:8])
        print("In myclient:152,Compute D1:",d1[:8])
        return d1
    
    def receive_updated_weights(self, server_flat_para) -> bool:
        '''with open("./testLog/list_file3.txt", "w") as file:
        # 将列表转换为字符串并写入文件
            file.write(str(server_flat_para))'''
        # Convert list of python integers into list of np.64
        server_flat_bias = list(np.array(server_flat_para[:(1<<13)], dtype=np.float32))
        server_flat_weights = server_flat_para[(1<<13):]
        # self.flat_params = server_flat_weights

        '''if self.flat_params is None:
            # first round (server gives full weights)
            self.flat_params = server_flat_weights
        else:
            # next rounds (server gives only gradient)
            self.flat_params = list(np.array(self.flat_params) + np.array(server_flat_weights))'''
        
        # Remove padding and return weights to original tensor structure and set model weights
        #这个remove_padding好像没起到什么实际作用，后面unflatten会自动忽略掉多余的参数？  
        #server_flat_weights = utils.remove_padding(self.flat_params,self.model_length)
        # Restore the long list of weights into the neural network's original structure
        #server_weights = utils.unflatten_weights(self.flat_params, self.model_shape)
        server_bias = utils.unflatten_weights(server_flat_bias, self.model_shape)
        server_weights = utils.unflatten_weights(server_flat_weights,self.weight_shape)
        server_weightspart = server_weights[:len(self.model_shape)]
        server_meaningpart = server_weights[len(self.model_shape):]
        print(f"In myclient:175,Fedavg plaintext(para or gra): {server_flat_bias[:8]}")

        self.set_parameters(server_bias,server_weightspart,server_meaningpart)
        #torch.save(self.model.state_dict(), 'D:/Graduation project/Code/FlowerBNN/testLog/Model3.pth')
        self.test_server_model()
    
    def test_server_model(self):
        '''y_pred = self.model.model.predict(X_test)
        predicted = np.argmax(y_pred, axis=-1)
        accuracy = np.equal(y_test, predicted).mean()
        loss = log_loss(y_test, y_pred)

        precision = precision_score(y_test, predicted)
        recall = recall_score(y_test, predicted)
        f1_score_ = f1_score(y_test, predicted)
        confusion_matrix_ = confusion_matrix(y_test, predicted)
        
        y_pred = self.model.model.predict(X_val)
        predicted = np.argmax(y_pred, axis=-1)
        val_accuarcy = np.equal(y_val, predicted).mean()

        print()
        print(f"\nLen(X_test): {len(X_test)}")
        print(f"Accuracy: {accuracy}")
        print(f"Val Accuarcy: {val_accuarcy}")
        print("Precision:", precision)
        print("Recall:",recall)
        print("F1-Score:", f1_score_)
        print(f"Loss: {loss}")
        print("\nConfusion matrix")
        print(confusion_matrix_)
        print()'''
        #上面这些测试指标有点点复杂,先搞一个Loss和正确率的，后面再添加
        Loss,Acc = test(self.model,testloader,DEVICE)
        print('Test after update parameters from server: Average loss: {:.4f}, Accuracy: {:.2f}% '.format(Loss,Acc))


if __name__ == "__main__":
    print("In test client")
    WEIGHT_DECIMALS = 7
    client_num = 3
    rlwe = Init_RLWE(WEIGHT_DECIMALS,client_num)
    
    #源码这里num_weights就是用来确定参数量的，暂时不用。但是展平等函数之后要实现
    fl.client.start_numpy_client(
         server_address="localhost:8080",
         client = MNISTClient(rlwe,WEIGHT_DECIMALS)
    )
