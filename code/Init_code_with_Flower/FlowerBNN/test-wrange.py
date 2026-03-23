import torch
from Net import *
from binarized_modules import Binarize,binarized
import numpy as np
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def data_num(nums,Max=20):
    Min_num = min(nums)
    for i in range(len(nums)):
        nums[i] = min(Max,nums[i]//Min_num)
    return nums

def Get_weight_range():
    testnet = Binary_CF_Mini().to(DEVICE)
    Modelname = '_BCFMini'
    mdpth = r'D:\Graduation project\Code\FlowerBNN\results\models\\'[:-1]
    for i in range(5,30,5):
        testnet_dict = torch.load(mdpth+Modelname+'{}.pth'.format(i))
        testnet.load_state_dict(testnet_dict)
        
        params = list(testnet.parameters())

        Max,Min = None,None
        for para in params:
            p_max = torch.max(para)
            p_min = torch.min(para)
            #print("type!",p_max.dtype,para.dtype)
            if Max is None or p_max>Max:
                Max = p_max
            if Min is None or p_min<Min:
                Min = p_min
        print("Get_Model_weight_range "+Modelname+"{},Max:{},Min:{}".format(i,Max,Min))


def Load_para():
    testnet = Binary_CF_Mini().to(DEVICE)
    #summary(testnet,(1,28,28))
    Modelname = 'BCFMini'
    mdpth = r'D:\Graduation project\Code\FlowerBNN\results\models\\'[:-1]
    
    testnet_dict = torch.load(mdpth+Modelname+'14.pth')
    testnet.load_state_dict(testnet_dict)
    return testnet

def get_parameters(net):
        #以 NumPy ndarrays 列表形式返回模型参数
        return [val.cpu().numpy() for _, val in net.state_dict().items()]

def get_binary_parameters(net):
        ret = []
        for name, val in net.state_dict().items():
            if 'weight' in name:
                print(name)
                ret.append(binarized(val).cpu().numpy())
                print(ret[-1])
            else:
                ret.append(val.cpu().numpy())
        cntw,cntb = 0,0
        '''for name, val in net.state_dict().items():
            if 'weight' in name :
                cntw += np.size(val.cpu().numpy())
                
            elif 'bias' in name:
                cntb += np.size(val.cpu().numpy())'''
                
        print(cntw,cntb)
        return ret
def get_weights(net):
    weights = []
    bias = []
    for name, val in net.state_dict().items():
        
        if 'weight' in name:
            weights.append(binarized(val).cpu().numpy())
            #print("WEIGHT",weights[-1].shape)
        elif 'bias' in name:
            bias.append(val.cpu().numpy())
        print('name:',name,val.shape)
    net.bias = bias
    return weights 
def set_parameters(net,parameters):
        #用从服务器接收到的参数更新本地模型参数
        params_dict = zip(net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        net.load_state_dict(state_dict, strict=True)

def Compare_Model(name1,name2):
    pth = 'D:/Graduation project/Code/FlowerBNN/testLog/'
    model1 = torch.load(pth+name1)
    # 加载第二个.pth文件
    model2 = torch.load(pth+name2)

    # 对比模型参数是否相同
    #parameters_match = all([torch.allclose(model1[key], model2[key]) for key in model1.keys()])
    for key in model1.keys():
        
        if torch.allclose(model1[key], model2[key]) == 0:
            print("怎么不一样！",key)
            print(model1[key])
           # print('\n\n',model2[key])
           # break

    '''if parameters_match:
        print("两个模型的参数完全相同。")
    else:
        print("两个模型的参数不完全相同。")
'''

if __name__ == '__main__':
    #Get_weight_range()
    #test_data = [20000,30000,5000,15000,65400]
    #print(data_num(test_data,10))

    #net = Load_para()
    #get_weights(net)
    #print("\nAHA\n")
    #print(net.bias)
    #pa = get_binary_parameters(net)
    '''for i in pa:
        print(i)'''
    #set_parameters(net,pa)
    name1 = 'Model1.pth'
    name2 = 'Model2.pth'
    Compare_Model(name1,name2)

    