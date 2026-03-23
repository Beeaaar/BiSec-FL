import torch
import Load_data
import torchvision.transforms as transforms
import torch.nn.functional as F
from torchvision.datasets import FashionMNIST
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
from Load_data import Load_part_data
from Modules import *
import os
# 计算指标
def evaluate_model(model, test_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    confusion_matrix_sum = None
    tp5 = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.topk(outputs, 5, dim=1)
            tp5 += sum([label in pre for label,pre in zip(labels,predicted)])
            # 计算混淆矩阵
            batch_confusion_matrix = confusion_matrix(labels.cpu(), predicted[:, 0].cpu(),labels=range(10))
            if confusion_matrix_sum is None:
                confusion_matrix_sum = batch_confusion_matrix
            else:
                confusion_matrix_sum += batch_confusion_matrix
    
    # 计算总体指标
    tp = confusion_matrix_sum.diagonal()
    fp = confusion_matrix_sum.sum(axis=0) - tp
    fn = confusion_matrix_sum.sum(axis=1) - tp
    tn = confusion_matrix_sum.sum() - (tp + fp + fn)
   
    top1_accuracy = tp.sum() / confusion_matrix_sum.sum()
    top5_accuracy = tp5 / confusion_matrix_sum.sum()
    print(tp,tp5)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    micro_f1 = 2 * (precision * recall) / (precision + recall)
    micro_f1 = micro_f1.mean()

    return top1_accuracy, top5_accuracy, precision.mean(), recall.mean(), micro_f1

# 加载模型
def load_model(model_path,model_name):
    if 'FMLP' in model_name:
        model = MLP()
    elif 'BMLP' in model_name:
        model = Binary_MLP()
    elif 'FCNN2' in model_name:
        model = CNN2()
    elif 'BCNN2' in model_name:
        model = Binary_CNN2()
    elif 'FCNN1' in model_name:
        model = CNN1()
    else:
        model = Binary_CNN1()
    
    model_dict = torch.load(model_path)
    model.load_state_dict(model_dict)
    return model

def Matrix(Name):
    model = Binary_CNN2(classes=10,WEIGHT_DECIMALS=7)
    path = "D:/Graduation project/Code/FlowerBNN/results/testall"
    modelpth = path+'/{}.pth'.format(Name)
    model_dict = torch.load(modelpth)
    model.load_state_dict(model_dict)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    _,test_loader = Load_part_data('FashionMNIST','niid3',idx=0,train_batch_size=32)
    confusion_matrix_sum = None
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.topk(outputs, 5, dim=1)
            #tp5 += sum([label in pre for label,pre in zip(labels,predicted)])
            # 计算混淆矩阵
            batch_confusion_matrix = confusion_matrix(labels.cpu(), predicted[:, 0].cpu(),labels=range(10))
            if confusion_matrix_sum is None:
                confusion_matrix_sum = batch_confusion_matrix
            else:
                confusion_matrix_sum += batch_confusion_matrix
    print(confusion_matrix_sum)
# 主函数
def main():
    Matrix('FashionBBCNN2')
    '''model_paths = [
        "D:/Graduation project/Code/FlowerBNN/results/testall/FashionFFMLP.pth",
        "D:/Graduation project/Code/FlowerBNN/results/testall/FashionBBMLP.pth",
        "D:/Graduation project/Code/FlowerBNN/results/testall/FashionFFCNN2.pth",
        "D:/Graduation project/Code/FlowerBNN/results/testall/FashionBBCNN2.pth",
        "D:/Graduation project/Code/FlowerBNN/results/testall/FashionFFCNN1.pth",
        "D:/Graduation project/Code/FlowerBNN/results/testall/FashionBBCNN1.pth"
    ]

    _,testloader = Load_part_data('FashionMNIST','niid3',idx=0,train_batch_size=32)

    with open("./results/testall/model_evaluation_results.txt", "w") as file:
        for model_path in model_paths:
            model_name = model_path.split("/")[-1].split(".")[0]
            model = load_model(model_path,model_name)
            top1_acc, top5_acc, prec, rec, f1 = evaluate_model(model, testloader)
            file.write(f"Model: {model_name}\n")
            file.write(f"Top1 Accuracy: {top1_acc * 100:.2f}\n")
            file.write(f"Top5 Accuracy: {top5_acc * 100:.2f}\n")
            file.write(f"Precision: {prec * 100:.2f}\n")
            file.write(f"Recall: {rec * 100:.2f}\n")
            file.write(f"Micro F1: {f1 * 100:.2f}\n\n")'''

if __name__ == "__main__":
    main()