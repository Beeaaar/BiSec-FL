#This is the client of FL with initial Flower
#主要是flwr和pytorch相关的包
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchsummary import summary
import Load_data
import time 
import argparse
from binarized_modules import Binarize,binarized
import Split_MNIST
import flwr as fl

from Modules import MLP,Binary_MLP,CNN1,Binary_CNN1,CNN2,Binary_CNN2,train_model,test_model
#分配设备
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


parser = argparse.ArgumentParser(description='Myclient Datapath')

parser.add_argument('--dataidx',metavar='DATA_IDX',
                    default=0,
                    help = 'data_idx')

parser.add_argument('--epLst', metavar='N', type=int, nargs='+',
                        help='an integer list')

args = parser.parse_args()
#datapth = 'D:/Graduation project/Code/FlowerBNN/data/MNIST/client1'
trainloader,testloader = Load_data.Load_part_data('MNIST','niid3',idx=args.dataidx,train_batch_size=32)
#trainloader,testloader = Load_data.Load_FEMNIST(idx=args.dataidx,train_batch_size=128,test_batch_size=64)
net = Binary_CNN2(classes=10).to(DEVICE)

#这一部分跟通信协议有点关系，如果消息格式改了，相应的参数更新和传递也要改吧
class MnistClient(fl.client.NumPyClient):
    def __init__(self,rnd=0):
        self.rnd = rnd
    def get_parameters(self, config):
        #以 NumPy ndarrays 列表形式返回模型参数
        return [val.cpu().numpy() for _, val in net.state_dict().items()]
    def get_binary_parameters(self,config):
        ret = []
        for name, val in net.state_dict().items():
            if 'weight' in name:
                ret.append(binarized(val).cpu().numpy())
            else:
                ret.append(val.cpu().numpy())
            #print(ret[-1])
        
        return ret
        return [binarized(val).cpu().numpy() for _, val in net.state_dict().items()]
    def set_parameters(self, parameters):
        #用从服务器接收到的参数更新本地模型参数
        params_dict = zip(net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        net.load_state_dict(state_dict, strict=True)

    #训练
    def fit(self, parameters, config):
        self.set_parameters(parameters)
        
        #print("Before Fit ")
        #loss,acc = test_model(net,testloader)
        #print(self.get_binary_parameters(config={}))
        self.rnd += 1
        #train_model(net,trainloader,testloader,testloader,DEVICE,epochs=args.epLst[self.rnd%9],lr = 1e-3)
        #train_model(net,trainloader,testloader,testloader,DEVICE,epochs=(60-self.rnd)//20+1,lr = 1e-3)
        train_model(net,trainloader,testloader,testloader,DEVICE,epochs=1,lr = 3e-4)
        #print("\nFit in client Send binary val")
        print("In Client After Fit")
        #loss,acc = test_model(net,testloader)
        #print(self.get_parameters(config={}))
        return self.get_parameters(config={}), len(trainloader.dataset), {}
        #return self.get_binary_parameters(config={}), len(trainloader.dataset), {}
    #测试
    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        print("In client evaluate!!")
        loss, accuracy = test_model(net, testloader)
        with open('./results/0803/BCNN2_MNIST_NIID{}.txt'.format(args.dataidx), 'a') as file:
            file.write(str("{:.4f}".format(loss))+' '+str("{:.2f}".format(accuracy)) + '\n')
        return float(loss), len(testloader.dataset), {"accuracy": float(accuracy)}    


def testNet():
    testnet = Binary_MLP().to(DEVICE)
    #summary(testnet,(1,28,28))
    Modelname = '_BCFMini'
    train_model(testnet,trainloader,epochs=30,modelname=Modelname)
    mdpth = r'D:\Graduation project\Code\FlowerBNN\results\models\\'[:-1]
    '''for i in range(5,31,5):
        testnet_dict = torch.load(mdpth+Modelname+'{}.pth'.format(i))
        testnet.load_state_dict(testnet_dict)
        print("Test Model "+Modelname+"{}".format(i))
        test(testnet,testloader)
    '''

    '''for name, layer in net.named_modules():
        layer.register_forward_hook(hook_fn)
        layer.register_forward_hook(param_hook_fn)''' 
    '''for parameters in net.parameters():#打印出参数矩阵及值
        print(parameters)

    for name, parameters in net.named_parameters():#打印出每一层的参数的大小
       print(name, ':', parameters.size())
    '''
if __name__ == '__main__':
    
    #testNet()
    #得把dataloader和net放全局
    fl.client.start_numpy_client(server_address="localhost:8080", client=MnistClient())
