from torchvision import datasets,transforms
import torch
import gzip
import numpy as np
from torch.utils.data import DataLoader,random_split,TensorDataset,ConcatDataset
import os
import json


datapth = r"D:\Graduation project\Code\FlowerBNN\data"


def Load_full_MNIST(validation = 0,train_batch_size=64,test_batch_size=32):
    #datapth = r"D:\Graduation project\Code\FlowerBNN\data"
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,),(0.3081,))
    ])

    #注意提前下载的数据集必须放在datapth下面/MNIST/raw,不然这个库函数找不着数据
    train_set = datasets.MNIST(root=datapth,train=True,transform=transform)
    test_set = datasets.MNIST(root=datapth,train=False,transform=transform)
    if validation:
        totdata = len(train_set)
        train_size = int(0.84*totdata)
        val_size = totdata-train_size
        train_set,val_set = random_split(train_set,[train_size,val_size])
    #然后根据batch_size参数处理成DataLoader类型,这里测试数据分批的时候没开随机化，每次分出来都是一样的
        val_loader = torch.utils.data.DataLoader(val_set,batch_size=test_batch_size,shuffle=False)
    train_loader = torch.utils.data.DataLoader(train_set,batch_size=train_batch_size,shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set,batch_size=test_batch_size,shuffle=False)

    if validation:
        return [train_loader,val_loader,test_loader] 
    else:
        return train_loader,test_loader


def Load_full_Fashion(validation = 0,train_batch_size=64,test_batch_size=32):   
    #datapth = r"D:\Graduation project\Code\FlowerBNN\data"
    train_set = datasets.FashionMNIST(root=datapth, train=True, transform=transforms.Compose([transforms.ToTensor()]) )
    test_set  = datasets.FashionMNIST(root=datapth, train=False, transform=transforms.Compose([transforms.ToTensor()]))
    if validation:
        totdata = len(train_set)
        train_size = int(0.84*totdata)
        val_size = totdata-train_size
        train_set,val_set = random_split(train_set,[train_size,val_size])
    #然后根据batch_size参数处理成DataLoader类型,这里测试数据分批的时候没开随机化，每次分出来都是一样的
        val_loader = torch.utils.data.DataLoader(val_set,batch_size=test_batch_size,shuffle=False)
    train_loader = torch.utils.data.DataLoader(train_set,batch_size=train_batch_size,shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set,batch_size=test_batch_size,shuffle=False)

    if validation:
        return [train_loader,val_loader,test_loader] 
    else:
        return train_loader,test_loader
    
def Load_full_EMNIST(validation = 0,train_batch_size=128,test_batch_size=64):
    
    train_set = datasets.EMNIST(root=datapth, train=True,transform=transforms.Compose([transforms.ToTensor()]),split='balanced')
    test_set  = datasets.EMNIST(root=datapth, train=False,transform=transforms.Compose([transforms.ToTensor()]),split='balanced')
    if validation:
        totdata = len(train_set)
        train_size = int(0.84*totdata)
        val_size = totdata-train_size
        train_set,val_set = random_split(train_set,[train_size,val_size])
    #然后根据batch_size参数处理成DataLoader类型,这里测试数据分批的时候没开随机化，每次分出来都是一样的
        val_loader = torch.utils.data.DataLoader(val_set,batch_size=test_batch_size,shuffle=False)
    train_loader = torch.utils.data.DataLoader(train_set,batch_size=train_batch_size,shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set,batch_size=test_batch_size,shuffle=False)

    if validation:
        return [train_loader,val_loader,test_loader] 
    else:
        return train_loader,test_loader

def Split_by_label(set,classes):
    ret = {}
    for data in set:
        if ret.get(data[1]):
            ret[data[1]].append(data)
        else:
            ret[data[1]] = [data]
    return ret
def Save_split_data(dataLst,split_num,savepth):
    for i in range(split_num):
        print(len(dataLst[i]))
        torch.save(dataLst[i], savepth+r'\train_dataset{}.pth'.format(i))
def Split_iid_data(name,split_num,split_lst,savepth):
    classes = 10
    if name == 'MNIST':
        train_set = datasets.MNIST(root=datapth,train=True,transform=transforms.Compose([transforms.ToTensor()]))
    elif name == 'FashionMNIST':
        train_set = datasets.FashionMNIST(root=datapth, train=True, transform=transforms.Compose([transforms.ToTensor()]) )
    else:
        train_set = datasets.EMNIST(root=datapth, train=True,transform=transforms.Compose([transforms.ToTensor()]),split='balanced' )
        classes = 47
    train_set = Split_by_label(train_set,classes)
    ret = [[] for _ in range(split_num)]
    totpart = sum(split_lst)
    for key,val in train_set.items():
        idx = 0
        tot = len(val)
        #print(tot)
        for j in range(split_num):
            idxnxt = idx + int(tot*split_lst[j]/totpart)
            ret[j] += val[idx:idxnxt]
            idx = idxnxt
            #print(key,j,idx)
    Save_split_data(ret,split_num,savepth)
def Split_niid_data(name,split_num,split_lst,savepth):
    classes = 10
    if name == 'MNIST':
        train_set = datasets.MNIST(root=datapth,train=True,transform=transforms.Compose([transforms.ToTensor()]))
    elif name == 'FashionMNIST':
        train_set = datasets.FashionMNIST(root=datapth, train=True, transform=transforms.Compose([transforms.ToTensor()]))
    else:
        train_set = datasets.EMNIST(root=datapth, train=True,transform=transforms.Compose([transforms.ToTensor()]),split='balanced' )
        classes = 47
    train_set = Split_by_label(train_set,classes)
    ret = [[] for _ in range(split_num)]
    i = -1
    for key,val in train_set.items():
        idx = 0
        i += 1
        tot = len(val)
        #print(tot)
        for j in range(split_num):
            idxnxt = idx + int(tot*split_lst[i][j])
            
            ret[j] += val[idx:idxnxt]
            idx = idxnxt
            #print(key,j,idx)
    Save_split_data(ret,split_num,savepth)

def Load_part_data(Name,method,idx,train_batch_size=64,test_batch_size=32):
    pth = datapth+r'\{}\{}\train_dataset{}.pth'.format(Name,method,idx)
    ret_train = torch.utils.data.DataLoader(torch.load(pth),batch_size=train_batch_size,shuffle=True)
    

    if Name == 'MNIST':
        test_set = datasets.MNIST(root=datapth,train=False,transform=transforms.Compose([transforms.ToTensor()]))
    elif Name == 'FashionMNIST':
        test_set  = datasets.FashionMNIST(root=datapth, train=False, transform=transforms.Compose([transforms.ToTensor()]))
    else:
        test_set  = datasets.EMNIST(root=datapth, train=False,transform=transforms.Compose([transforms.ToTensor()]),split='letters')
    ret_test = torch.utils.data.DataLoader(test_set,batch_size=test_batch_size,shuffle=False)
    return ret_train,ret_test

def Deal_with_FEMNIST(numLst,datapth,savepth):
    # 1. 加载JSON文件
    for rawnum in numLst:
        file_name = os.path.join(datapth,'all_data_{}.json'.format(rawnum))
        with open(file_name, 'r') as f:
            data = json.load(f)

        # 2. 整理数据
        images = []
        labels = []
        for user in data['users']:
            user_data = data['user_data'][user]
            num_samples = data['num_samples'][data['users'].index(user)]
            for i in range(num_samples):
                image = user_data['x'][i]
                label = user_data['y'][i]
                images.append(image)
                labels.append(label)

        # 3. 划分训练集和测试集
        images = np.array(images[:150000])
        labels = np.array(labels[:150000])

        images = images.reshape(-1, 1, 28, 28)
        X_tensor = torch.tensor(images, dtype=torch.float32)
        y_tensor = torch.tensor(labels, dtype=torch.long)

        # 创建TensorDataset
        dataset = TensorDataset(X_tensor, y_tensor)

        # 划分训练集和测试集
        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

        torch.save(train_dataset, os.path.join(savepth,'train_dataset{}.pth'.format(rawnum)))
        torch.save(test_dataset, os.path.join(savepth,'test_dataset{}.pth'.format(rawnum)))

def Load_FEMNIST(idx = '0',train_batch_size=64,test_batch_size=32):
    datapth = r'D:\Graduation project\Code\FlowerBNN\data\FEMNIST'
    train_dataset = torch.load(os.path.join(datapth,'train_dataset{}.pth'.format(idx)))
    #test_dataset = torch.load(os.path.join(datapth,'test_dataset{}.pth'.format(idx)))
    test_dataset = torch.load(os.path.join(datapth,'test_dataset{}.pth'.format(idx)))
    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)

    '''data_iter = iter(train_loader)
    images, labels = next(data_iter)

    # 输出数据批次的维度
    print("Images shape:", images.shape)
    print("Labels shape:", labels.shape)'''
    return train_loader,test_loader
def Combine_test(idxLst=[0,1,2],datapth=r'D:\Graduation project\Code\FlowerBNN\data\FEMNIST'):
    testset = []
    for idx in idxLst:
        print(idx)
        pth = os.path.join(datapth,'test_dataset{}.pth'.format(idx))
        testset.append(torch.load(pth))
    combined_dataset = ConcatDataset(testset)
    torch.save(combined_dataset, os.path.join(datapth,'test_dataset.pth'))
def Combine_train(idxLst=[0,1,2],datapth=r'D:\Graduation project\Code\FlowerBNN\data\FEMNIST'):
    trainset = []
    for idx in idxLst:
        print(idx)
        pth = os.path.join(datapth,'train_dataset{}.pth'.format(idx))
        trainset.append(torch.load(pth))
    combined_dataset = ConcatDataset(trainset)
    torch.save(combined_dataset, os.path.join(datapth,'train_dataset012.pth'))
if __name__ == '__main__':
    '''ratio = [[0.85,0.05,0.1],
             [0.65,0.15,0.2],
             [0.7,0.25,0.05],
             [0.8,0.1,0.1],
             [0.45,0.35,0.2],
             [0.05,0.85,0.1],
             [0.2,0.67,0.13],
             [0.15,0,2,0.65],
             [0.2,0.1,0.7],
             [0.2,0.05,0.75]]
    Split_niid_data('FashionMNIST',3,ratio,r'D:\Graduation project\Code\FlowerBNN\data\FashionMNIST\niid3')
    Split_iid_data('FashionMNIST',3,[3,2,1],r'D:\Graduation project\Code\FlowerBNN\data\FashionMNIST\iid3')'''
    #Deal_with_FEMNIST([3,4,5],r'D:\Graduation project\Code\FlowerBNN\data\leaf-master\data\femnist\data\all_data',r'D:\Graduation project\Code\FlowerBNN\data\FEMNIST')
    #Combine_test(idxLst=[3,4,5])
    Combine_train(idxLst=[0,1,2,3,4,5])