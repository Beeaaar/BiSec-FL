import torch
import torch.nn as nn
from binarized_modules import  BinarizeLinear,BinarizeConv2d
from torchsummary import summary
import torch.optim as optim
import time
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict
#和平时网络的区别在于1.全连接层是二值化的（也有二值化的卷积，但是这个网络没用到），2.激活函数限制了输出在【-1，1】之间（对应论文要求）
class Binary0(nn.Module):
    def __init__(self):
        
        super(Binary0,self).__init__()
        self.infl_ratio = 3

        self.fc1 = BinarizeLinear(784, 1024*self.infl_ratio)
        self.htanh1 = nn.Hardtanh()
        self.bn1 = nn.BatchNorm1d(1024*self.infl_ratio)
        self.fc2 = BinarizeLinear(1024*self.infl_ratio, 2048*self.infl_ratio)
        self.htanh2 = nn.Hardtanh()
        self.bn2 = nn.BatchNorm1d(2048*self.infl_ratio)
        self.fc3 = BinarizeLinear(2048*self.infl_ratio, 2048*self.infl_ratio)
        self.htanh3 = nn.Hardtanh()
        self.bn3 = nn.BatchNorm1d(2048*self.infl_ratio)
        self.fc4 = nn.Linear(2048*self.infl_ratio, 10)
        self.logsoftmax=nn.LogSoftmax()
        #Dropout（0.5）表示神经元有0.5的概率不会被激活（置为0），一般是为了防止过拟合
        self.drop=nn.Dropout(0.5)
        
    def forward(self,x):
        #view类似于reshape，-1表示动态调整这个维度的大小来保证x总元素个数前后一致。这里MNIST数据集正常来说就是28*28的
        x = x.view(-1, 28*28)
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.htanh1(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.htanh2(x)
        x = self.fc3(x)
        #在第三个全连接层后面drop了一下
        x = self.drop(x)
        x = self.bn3(x)
        x = self.htanh3(x)
        x = self.fc4(x)
        return self.logsoftmax(x)


class CF(nn.Module):
    def __init__(self,num_classes= 10):
        super(CF,self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1,48,3,padding=1),
            nn.MaxPool2d(kernel_size=2,stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(48),

            nn.Conv2d(48,128,3,padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),

            nn.Conv2d(128,256,3,padding=1),
            nn.MaxPool2d(kernel_size=2,stride=1,padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
        )
        self.infl_ratio = 3
        self.classifier = nn.Sequential(
            nn.Linear(256*15*15,1024*self.infl_ratio),
            nn.BatchNorm1d(1024*self.infl_ratio),
            nn.ReLU(),

            nn.Linear(1024*self.infl_ratio,2048*self.infl_ratio),
            nn.BatchNorm1d(2048*self.infl_ratio),
            nn.ReLU(),

            nn.Linear(2048*self.infl_ratio,num_classes),
            nn.LogSoftmax(),
            nn.Dropout(0.3)
        )
    
    def forward(self,x):
        x = self.features(x)
        x = x.view(-1,256*15*15)
        x = self.classifier(x)
        return x
        
class Binary_CF(nn.Module):
    def __init__(self,num_classes=10):
        super(Binary_CF,self).__init__()
        self.features = nn.Sequential(
            BinarizeConv2d(1,48,3,padding=1),
            nn.MaxPool2d(kernel_size=2,stride=2),
            nn.BatchNorm2d(48),
            nn.Hardtanh(),

            BinarizeConv2d(48,128,3,padding=1),
            nn.BatchNorm2d(128),
            nn.Hardtanh(),

            BinarizeConv2d(128,256,3,padding=1),
            nn.MaxPool2d(kernel_size=2,stride=1,padding=1),
            nn.BatchNorm2d(256),
            nn.Hardtanh(),
        )
        self.infl_ratio = 3
        self.classifier = nn.Sequential(
            BinarizeLinear(256*15*15,1024*self.infl_ratio),
            nn.BatchNorm1d(1024*self.infl_ratio),
            nn.Hardtanh(),

            BinarizeLinear(1024*self.infl_ratio,2048*self.infl_ratio),
            nn.BatchNorm1d(2048*self.infl_ratio),
            nn.Hardtanh(),

            BinarizeLinear(2048*self.infl_ratio,num_classes),
            nn.LogSoftmax(),
            nn.Dropout(0.3)
        )

    def forward(self,x):
        x = self.features(x)
        x = x.view(-1,256*15*15)
        x = self.classifier(x)
        return x

class CF_Mini(nn.Module):
    def __init__(self,num_classes= 10):
        super(CF_Mini,self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1,32,3,padding=1),
            nn.MaxPool2d(kernel_size=2,stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(32),

            nn.Conv2d(32,64,3,padding=1),
            nn.MaxPool2d(kernel_size=2,stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(64),
        )
        self.infl_ratio = 3
        self.classifier = nn.Sequential(
            nn.Linear(64*7*7,1024*self.infl_ratio),
            nn.BatchNorm1d(1024*self.infl_ratio),
            nn.ReLU(),

            nn.Linear(1024*self.infl_ratio,num_classes),
            nn.LogSoftmax(),
            nn.Dropout(0.3)
        )
    
    def forward(self,x):
        x = self.features(x)
        x = x.view(-1,64*7*7)
        x = self.classifier(x)
        return x

class Binary_CF_Mini(nn.Module):
    def __init__(self,num_classes= 10,WEIGHT_DECIMALS=6):
        self.WEIGHT_DECIMALS = WEIGHT_DECIMALS if WEIGHT_DECIMALS<=7 else 7
        super(Binary_CF_Mini,self).__init__()
        self.features = nn.Sequential(
            BinarizeConv2d(1,32,3,padding=1),
            nn.MaxPool2d(kernel_size=2,stride=2),
            nn.BatchNorm2d(32),
            nn.Hardtanh(),

            BinarizeConv2d(32,64,3,padding=1),
            nn.MaxPool2d(kernel_size=2,stride=2),
            nn.BatchNorm2d(64),
            nn.Hardtanh(),
        )
        self.infl_ratio = 3
        self.classifier = nn.Sequential(
            BinarizeLinear(64*7*7,1024*self.infl_ratio),
            nn.BatchNorm1d(1024*self.infl_ratio),
            nn.Hardtanh(),

            BinarizeLinear(1024*self.infl_ratio,num_classes),
            nn.LogSoftmax(dim=-1),
            nn.Dropout(0.3)
        )
    
    def forward(self,x):
        x = self.features(x)
        x = x.view(-1,64*7*7)
        x = self.classifier(x)
        return x
    
class Binary_CF_superMini(nn.Module):
    def __init__(self,num_classes= 10,WEIGHT_DECIMALS=6):
        self.WEIGHT_DECIMALS = WEIGHT_DECIMALS if WEIGHT_DECIMALS<=7 else 7
        super(Binary_CF_superMini,self).__init__()
        self.features = nn.Sequential(
            BinarizeConv2d(1,32,3,padding=1),
            nn.MaxPool2d(kernel_size=2,stride=2),
            nn.BatchNorm2d(32),
            nn.Hardtanh(),
        )
        self.infl_ratio = 2
        self.classifier = nn.Sequential(
            BinarizeLinear(32*14*14,1024*self.infl_ratio),
            nn.BatchNorm1d(1024*self.infl_ratio),
            nn.Hardtanh(),

            nn.Linear(1024*self.infl_ratio,num_classes),
            nn.LogSoftmax(dim=-1),
            nn.Dropout(0.5)
        )
    
    def forward(self,x):
        x = self.features(x)
        x = x.view(-1,32*14*14)
        x = self.classifier(x)
        return x
        
    
def get_weights(model):
    weights = model.parameters()

    #[val.cpu().numpy() for _, val in self.model.state_dict().items()]
    scaled_weights = [torch.round(para.detach().cpu()*(10**model.WEIGHT_DECIMALS)) for para in weights]
    return scaled_weights
def get_bias(model):
    scaled_bias = []
    for name,para in model.named_parameters():
        if "bias" in name:
    #[val.cpu().numpy() for _, val in self.model.state_dict().items()]
            scaled_bias.append(torch.round(para.detach().cpu()*(10**model.WEIGHT_DECIMALS))) 
    return scaled_bias

def set_weight(model,scaled_weights):
    weights = [para/(10**model.WEIGHT_DECIMALS) for para in scaled_weights]
    for model_para,weight in zip(model.parameters(),weights):
        model_para.data.copy_(weight)

'''def set_bias(model,scaled_bias):
    bias = [para/(10**model.WEIGHT_DECIMALS) for para in scaled_bias]
    j = 0
    for name,para in model.named_parameters():

        if "bias" in name:
            para.data.copy_(bias[j])
            if j <= 2:
                print("In Net,Set_bias",name,bias[j][:4])
            j+=1'''


def set_weight_and_bias(model,weights,scaled_bias):
    bias = [sb/(10**model.WEIGHT_DECIMALS) for sb in scaled_bias]
    #print(type(bias[0]))
    #print(type(weights[0]))
    i,j = 0,0
    for name,para in model.named_parameters():
        if 'weight' in name:
            para.data.copy_(weights[i])
            i+=1
        else:
            para.data.copy_(bias[j])
            j+=1
def set_bias(model,scaled_bias):
    bias = [sb/(10**model.WEIGHT_DECIMALS) for sb in scaled_bias]
    #print(type(bias[0]))
    #print(type(weights[0]))
    i = 0
    for name,para in model.named_parameters():
        if 'bias' in name:
            para.data.copy_(bias[i])
            i+=1
def set_weight_and_meaning(model,para):
    i = 0
    for name, val in model.state_dict().items():
        if 'num_batches_tracked' in name or 'bias' in name: continue
        val.data.copy_(para[i])
        i+=1

def set_weight_and_bias_and_meaning(model,scaled_bias,weights,scaled_meaning):
    i,j,k = 0,0,0
    bias = [sb/(10**model.WEIGHT_DECIMALS) for sb in scaled_bias]
    meaning = [sm/(10000) for sm in scaled_meaning]
    for name, val in model.state_dict().items():
        if 'num_batches_tracked' in name: continue
        if 'bias' in name:
            print("IN NET set bias:",bias[i][:8])
            val.data.copy_(bias[i])
            i+=1
        elif 'weight' in name:
            #print("IN NET set weight:",weights[j][:4])
            val.data.copy_(weights[j])
            j+=1
        else:
            print("IN NET set meaning:",meaning[k][:8])
            val.data.copy_(meaning[k])
            k+=1         
def train(model,trainloader,DEVICE,epochs = 1,lr=1e-3,modelname = 'model'):
    model.train()
    optimizer = optim.Adam(model.parameters(),lr=lr)

    for ep in range(epochs):
        start = time.time()
        totLoss = 0
        #动态调整学习率，这个10和0.1都是超参
        if ep%10 == 0:
            #lr = lr*0.1
            optimizer.param_groups[0]['lr']=optimizer.param_groups[0]['lr']*0.1
        for imgs,lbs in trainloader:
            imgs,lbs = imgs.to(DEVICE),lbs.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = F.cross_entropy(outputs,lbs)
            totLoss += loss.item()
            loss.backward()

            #参数进行回滚，再更新，然后正则化限定范围（原理来自论文，再理解一下）
            for p in list(model.parameters()):
                if hasattr(p,'org'):
                    p.data.copy_(p.org)
            optimizer.step()
            for p in list(model.parameters()):
                if hasattr(p,'org'):
                    p.org.copy_(p.data.clamp_(-1,1))
        
        print('Train Epoch:{}\tLoss:{:.6f},time:{}s'.format(ep,totLoss/len(trainloader),time.time()-start))
        '''if (ep+1)%5 ==0:
            test(model,testloader)
            torch.save(model.state_dict(), 'D:/Graduation project/Code/FlowerBNN/results/models/'+modelname+'{}.pth'.format(ep))'''
        
def test(model,testloader,DEVICE):
    model.eval()
    correct = 0.0
    loss = 0.0

    with torch.no_grad():
        for imgs,lbs in testloader:
            imgs,lbs = imgs.to(DEVICE),lbs.to(DEVICE)
            output = model(imgs)
            loss += F.cross_entropy(output,lbs).item()
            predict = output.argmax(dim=1)
            correct += (predict==lbs).sum().item()
        #print("test_avarage_loss:{:.4f},accuracy:{:.4f}%".format(loss/total,100*(correct/total)))

    print('\nTest set: Average loss: {:.4f}, Accuracy: {:.2f}%'.format(loss/len(testloader.dataset),100*correct/len(testloader.dataset)))
    return loss/len(testloader.dataset),100*correct/len(testloader.dataset)

'''DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net = Binary_CF_superMini().to(DEVICE)
summary(net,(1,28,28),batch_size = 64)'''

def hook_fn(module, input, output):
    print(module)  # 打印当前层的信息
    print("Input:", input)  # 打印输入
    print("Output:", output)  # 打印输出

# 注册forward hook


def param_hook_fn(module, input, output):
    print(module)  # 打印当前层的信息
    for name, param in module.named_parameters():
        print(f"Parameter name: {name}, Parameter shape: {param.shape}")
        print(param.data)  # 打印参数数值

    
