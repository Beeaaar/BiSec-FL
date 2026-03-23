import numpy as np
from torchvision import datasets,transforms
import torch
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
    return train_set,test_set
def Split_by_ratio(label,ratio):
    ret =[[] for _ in range(3)]
    for i in range(10):
        idx = 0
        tot = len(label[i])
        for j in range(3):
            idxnxt = idx + int(tot*ratio[i][j])
            ret[j] += label[i][idx:idxnxt]
            idx = idxnxt

    for i in range(3):
        print(len(ret[i]))
    return ret

def Split_by_label(set):
    ret = [[] for _ in range(10)]
    for data in set:
        ret[data[1]].append(data)
    '''for i in range(10):
        print(len(ret[i]))'''
    return ret
def Save_data(dataLst):
    num = len(dataLst)
    print(num)
    for i in range(num):
        torch.save(dataLst[i], 'D:/Graduation project/Code/FlowerBNN/data/MNIST/part/train_dataset{}.pth'.format(i))
def Load_part_data(pth = 'D:/Graduation project/Code/FlowerBNN/data/MNIST/part/',batch_size = 64,idx=0):
    '''Load = []
    for i in range(3):
        Load.append(torch.load(pth+'train_dataset{}.pth'.format(i)))
        print('Load:',len(Load[i]))
        Load[i] = torch.utils.data.DataLoader(Load[i],batch_size=batch_size,shuffle=True)
    return Load'''
    ret_train = torch.load(pth+'train_dataset{}.pth'.format(idx))
    ret_train = torch.utils.data.DataLoader(ret_train,batch_size=batch_size,shuffle=True)
    _,test_set = load_data()
    ret_test = torch.utils.data.DataLoader(test_set,batch_size=batch_size//2,shuffle=False)
    return ret_train,ret_test
if __name__ == '__main__':
    print("SPLIT MNIST")
    '''train_set,test_set = load_data()
    split_by_label = Split_by_label(train_set)
    ratio = [[0.7,0.2,0.1],
             [0.65,0.15,0.2],
             [0.7,0.25,0.05],
             [0.8,0.1,0.1],
             [0.6,0.25,0.15],
             [0.1,0.8,0.1],
             [0.2,0.67,0.13],
             [0.15,0,2,0.65],
             [0.3,0.1,0.6],
             [0.2,0.05,0.75]]
    Three_dataset = Split_by_ratio(split_by_label,ratio)
    Save_data(Three_dataset)'''
    Load = Load_part_data()
