from torchvision import datasets,transforms
import torch
import gzip
import numpy as np
from torch.utils.data import DataLoader
def load_part_data(datapath):
    #按照mnist的数据集格式读入，转为tensor类型,读分割好的数据
    with gzip.open(datapath+r'/train_lbs.gz') as pth:
        lbs = np.frombuffer(pth.read(),dtype=np.uint8,offset=8)

    with gzip.open(datapath+r'/train_imgs.gz') as pth:
        imgs = np.frombuffer(pth.read(),dtype=np.uint8,offset=16).reshape(len(lbs),1,28,28)

    with gzip.open(datapath+r'/test_lbs.gz') as pth:
        tlbs = np.frombuffer(pth.read(),dtype=np.uint8,offset=8)

    with gzip.open(datapath+r'/test_imgs.gz') as pth:
        timgs = np.frombuffer(pth.read(),dtype=np.uint8,offset=16).reshape(len(tlbs),1,28,28)

    lbs = torch.from_numpy(lbs).type(torch.LongTensor)
    imgs = torch.from_numpy(imgs).type(torch.float32)
    tlbs = torch.from_numpy(tlbs).type(torch.LongTensor)
    timgs = torch.from_numpy(timgs).type(torch.float32)

    trainset = torch.utils.data.TensorDataset(imgs,lbs)
    testset = torch.utils.data.TensorDataset(timgs,tlbs)

    num = {"trainset" : len(trainset), "testset" : len(testset)}
    trainloader = DataLoader(trainset,batch_size=64,shuffle=True)
    testloader = DataLoader(testset,batch_size=32,shuffle=False)

    return trainloader,testloader

def load_data(datapth ='D:/Graduation project/Code/FlowerBNN/data',batch_size = 128,test_batch_size = 64):
    #数据处理，转成Tensor并且进行标准化处理（0.1307和0.3081是提前算好的数据集均值和标准差）
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,),(0.3081,))
    ])
    #调库加载数据集，能把训练的图片和标签都处理出来
    #注意提前下载的数据集必须放在datapth下面/MNIST/raw,不然这个库函数找不着数据
    train_set = datasets.MNIST(root=datapth,train=True,transform=transform)
    test_set = datasets.MNIST(root=datapth,train=False,transform=transform)
    #然后根据batch_size参数处理成DataLoader类型,这里测试数据分批的时候没开随机化，每次分出来都是一样的
    train_loader = torch.utils.data.DataLoader(train_set,batch_size=batch_size,shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set,batch_size=test_batch_size,shuffle=False)

    return train_loader,test_loader