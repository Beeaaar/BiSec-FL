#定义了最终使用的三个网络结构和他们的二值化
#定义了测试，训练，存储，加载模型，显示模型结构，计算参数数量等基础模型功能
import torch
import torch.nn as nn
from binarized_modules import  BinarizeLinear,BinarizeConv2d
from torchsummary import summary
import torch.optim as optim
import time
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict
import Load_data
import copy

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class MLP(nn.Module):
    def __init__(self,classes = 10):
        super(MLP,self).__init__()
        self.fc1 = nn.Linear(784,512)
        self.bn1 = nn.BatchNorm1d(512)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(512,128)
        self.bn2 = nn.BatchNorm1d(128)
        self.relu2 = nn.ReLU()
        self.drop=nn.Dropout(0.5)
        self.fc3 = nn.Linear(128,classes)
        self.softmax = nn.LogSoftmax(dim=1)
        
    def forward(self,x):
        x = x.view(-1,28*28)
        x = self.relu1(self.bn1(self.fc1(x)))
        x = self.relu2(self.bn2(self.fc2(x)))
        x = self.softmax(self.fc3(self.drop(x)))
        return x
    
class Binary_MLP(nn.Module):
    def __init__(self,classes = 10):
        
        super(Binary_MLP,self).__init__()
        self.fc1 = BinarizeLinear(784, 512)
        self.htanh1 = nn.Hardtanh()
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = BinarizeLinear(512, 128)
        self.drop = nn.Dropout(0.3)
        self.htanh2 = nn.Hardtanh()
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, classes)
        self.logsoftmax=nn.LogSoftmax(dim=1)
        
    def forward(self,x):
        x = x.view(-1, 28*28)
        x = self.htanh1(self.bn1(self.fc1(x)))
        x = self.htanh2(self.bn2(self.fc2(x)))
        x = self.logsoftmax(self.drop(self.fc3(x)))
        return x

class CNN1(nn.Module):
    def __init__(self,classes= 10):
        super(CNN1,self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1,48,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2),
            nn.BatchNorm2d(48),

            nn.Conv2d(48,128,3,padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),

            nn.Conv2d(128,256,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2),
            nn.BatchNorm2d(256),
        )
        self.classifier = nn.Sequential(
            nn.Linear(256*7*7,2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(2048,classes),
            nn.LogSoftmax(dim=1),
        )
    
    def forward(self,x):
        x = self.features(x)
        x = x.view(-1,256*7*7)
        x = self.classifier(x)
        return x
        

class Binary_CNN1(nn.Module):
    def __init__(self,classes= 10):
        super(Binary_CNN1,self).__init__()
        self.features = nn.Sequential(
            BinarizeConv2d(1,48,3,padding=1),
            nn.MaxPool2d(kernel_size=2,stride=2),
            nn.BatchNorm2d(48),
            nn.Hardtanh(),

            BinarizeConv2d(48,128,3,padding=1),
            nn.BatchNorm2d(128),
            nn.Hardtanh(),

            BinarizeConv2d(128,256,3,padding=1),
            nn.MaxPool2d(kernel_size=2,stride=2),
            nn.BatchNorm2d(256),
            nn.Hardtanh(),
        )
        self.classifier = nn.Sequential(
            BinarizeLinear(256*7*7,2048),
            nn.BatchNorm1d(2048),
            nn.Hardtanh(),
            nn.Dropout(0.5),
            nn.Linear(2048,classes),
            nn.LogSoftmax(dim=1),
        )
    def forward(self,x):
        x = self.features(x)
        x = x.view(-1,256*7*7)
        x = self.classifier(x)
        return x


class CNN2(nn.Module):
    def __init__(self,classes= 10,WEIGHT_DECIMALS=6):
        self.WEIGHT_DECIMALS = WEIGHT_DECIMALS if WEIGHT_DECIMALS<=7 else 7
        super(CNN2,self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1,32,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2),
            nn.BatchNorm2d(32),
        )
        self.classifier = nn.Sequential(
            nn.Linear(32*14*14,2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(2048,classes),
            nn.LogSoftmax(dim=1),  
        )
    def forward(self,x):
        x = self.features(x)
        x = x.view(-1,32*14*14)
        x = self.classifier(x)
        return x

class Binary_CNN2(nn.Module):
    def __init__(self,classes= 10,WEIGHT_DECIMALS=6):
        self.WEIGHT_DECIMALS = WEIGHT_DECIMALS if WEIGHT_DECIMALS<=7 else 7
        super(Binary_CNN2,self).__init__()
        self.features = nn.Sequential(
            BinarizeConv2d(1,32,3,padding=1),
            nn.MaxPool2d(kernel_size=2,stride=2),
            nn.BatchNorm2d(32),
            nn.Hardtanh(),
        )
        self.classifier = nn.Sequential(
            BinarizeLinear(32*14*14,2048),
            nn.BatchNorm1d(2048),
            nn.Hardtanh(),
            nn.Dropout(0.5),
            nn.Linear(2048,classes),
            nn.LogSoftmax(dim=1),
        )
    
    def forward(self,x):
        x = self.features(x)
        x = x.view(-1,32*14*14)
        x = self.classifier(x)
        return x


#这个函数好像暂时没什么用
'''def get_num_of_para(model,paras = ['']):
    ret = []
    for name, val in model.state_dict().items():
        for para in paras:
            if para in name:
                ret.append[val.cpu().numpy()]
    return ret'''

def save_model(model,modelname ='model'):
    pth = 'D:/Graduation project/Code/FlowerBNN/results/models/'
    torch.save(model.state_dict(), pth+modelname+'.pth')

def load_model(model,modelname='model'):
    pth = 'D:/Graduation project/Code/FlowerBNN/results/models/'
    model_dict = torch.load(pth+modelname+'.pth')
    model.load_state_dict(model_dict)

def train_model(model,trainloader,valoader,testloader,DEVICE=DEVICE,epochs = 10,lr=1e-3,modelname = 'MLP',Save = 0):
    model.train()
    model_to_save = None
    maxAcc = 0
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
        Loss,Acc = test_model(model,valoader,DEVICE)
        if Acc > maxAcc:
            maxAcc = Acc
            model_to_save = copy.deepcopy(model)
        #Loss,Acc = test_model(model,valoader,DEVICE)
    if Save:
        save_model(model_to_save,modelname)
            
def test_model(model,testloader,DEVICE=DEVICE):
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

    print('In test_model: Average loss: {:.4f}, Accuracy: {:.2f}%'.format(loss/len(testloader.dataset),100*correct/len(testloader.dataset)))
    return loss/len(testloader.dataset),100*correct/len(testloader.dataset)

def show_model_structure(model):
    summary(model,(1,28,28),batch_size = 32)
    for name, val in model.state_dict().items():
        print(name,val.shape)



if __name__ == '__main__':
    
    net = CNN1(classes=10).to(DEVICE)
    #model_name = 'BMLP_FashionMNIST'
    #trainloader,valoader,testloader = Load_data.Load_full_Fashion(validation=1)
    #trainloader,testloader = Load_data.Load_part_data('MNIST','iid3',idx=1,train_batch_size=32)
    #trainloader,testloader =trainloader,valoader,testloader = Load_data.Load_full_Fashion(validation=1) Load_data.Load_FEMNIST(idx='012',train_batch_size=128)
    show_model_structure(net)
    #valoader = testloader
    #train_model(net,trainloader,valoader,testloader,DEVICE,epochs=30,lr = 1e-3,modelname = model_name,Save=0)
    #load_model(net,model_name)
    #test_model(net,testloader,DEVICE)
    