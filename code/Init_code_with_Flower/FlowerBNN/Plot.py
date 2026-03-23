import matplotlib.pyplot as plt
import matplotlib
from matplotlib.font_manager import FontManager
import numpy as np
# 设置中文字体
#matplotlib.rc("font",family='KaiTi')
plt.rcdefaults()
matplotlib.rc("font",family='Microsoft YaHei')
'''mpl_fonts = set(f.name for f in FontManager().ttflist)
print('all font list get from matplotlib.font_manager:')
for f in sorted(mpl_fonts):
    print('\t' + f)'''

def Draw_Plot():

    file_path = 'D:/Graduation project/Code/FlowerBNN/results/AccMLP-MNIST5-5/'

    # 读取文件并提取第二列数据作为y坐标
    file1Name = ['AccFF','AccBF','AccFB','AccBB']
    file2Name = ['NAccFF','NAccBF','NAccFB','NAccBB']
    labelName = ['FF','BF','FB','BB']
    y1 = [[] for _ in range(len(file1Name))]
    y2 = [[] for _ in range(len(file2Name))]

    for idx,filename in enumerate(file1Name):

        with open(file_path+'{}.txt'.format(filename), 'r') as file:
            for line in file:
                # 如果文件中每行是以空格或制表符分隔的两列数据
                columns = line.split()  # 使用空格分隔数据
                if len(columns) >= 2:
                    y_value = float(columns[1])  # 获取第二列数据，转换为浮点数
                    y1[idx].append(y_value)

    for idx,filename in enumerate(file2Name):
        with open(file_path+'{}.txt'.format(filename), 'r') as file:
            for line in file:
                # 如果文件中每行是以空格或制表符分隔的两列数据
                columns = line.split()  # 使用空格分隔数据
                if len(columns) >= 2:
                    y_value = float(columns[1])  # 获取第二列数据，转换为浮点数
                    y2[idx].append(y_value)

    # 创建对应的x坐标
    x_data = list(range(1, len(y1[0]) + 1))  # 生成与y数据长度相同的序列作为x坐标

    # 创建图形和坐标轴对象
    #plt.figure(figsize=(16, 6))
    fig,ax = plt.subplots(1,2,figsize=(12, 5))
    plt.rcParams.update({'font.size': 16}) 
    # 绘制折线
    for i,y_data in enumerate(y1):
        ax[0].plot(x_data, y_data, label=labelName[i])
    #ax[0].set_xlabel('联邦学习聚合轮数',fontsize=16)
    #ax[0].set_ylabel('模型测试准确率',fontsize=16)
    ax[0].set_xlabel('Aggregation Round (IID)',fontsize=16)
    ax[0].set_ylabel('Test Accuracy(%)',fontsize=16)
    ax[0].set_ylim(70,100)
    ax[0].legend()
    

    for i,y_data in enumerate(y2):
        ax[1].plot(x_data, y_data, label=labelName[i])
    # 添加标题和标签
    #ax[1].set_xlabel('联邦学习聚合轮数', fontsize=16)
    #ax[1].set_ylabel('模型测试准确率', fontsize=16)
    ax[1].set_xlabel('Aggregation Round (NIID)', fontsize=16)
    ax[1].set_ylabel('Test Accuracy(%)', fontsize=16)
    ax[1].set_ylim(70,100)
    ax[1].legend()
    # 显示图形
    
    plt.show()

def Cmprate():
    file_path = 'D:/Graduation project/Code/FlowerBNN/results/AccMLP-MNIST5-5/'

    # 读取文件并提取第二列数据作为y坐标
    fileName = ['NAccBF','Cmp']
    y = [[] for _ in range(len(fileName))]

    for idx,filename in enumerate(fileName):
        with open(file_path+'{}.txt'.format(filename), 'r') as file:
            for line in file:
                # 如果文件中每行是以空格或制表符分隔的两列数据
                columns = line.split()  # 使用空格分隔数据
                if len(columns) >= 2:
                    y_value = float(columns[1])  # 获取第二列数据，转换为浮点数
                    y[idx].append(y_value)
        
    x_data = list(range(1,41))
    plt.figure(figsize=(8, 6))

    # 画折线图
    for i in range(len(y)):
        plt.plot(x_data, y[i][:40], label='Line {}'.format(i))
    plt.legend()
    plt.show()

def Draw_MNIST_MLPandCNN2():
    #plt.rcParams['font.family'] = ['KaiTi','SimSun','Microsoft YaHei', 'SimHei']
    file_path = 'D:/Graduation project/Code/FlowerBNN/results/0803/'

    # 读取文件并提取第二列数据作为y坐标
    fileName = ['MLP_MNIST','BMLP_MNIST','CNN2_MNIST','BCNN2_MNIST']
    labelName = ['MLP','BMLP','CNN2','BCNN2']
    #par = 'IID'
    par = 'NIID'
    y1 = [[] for _ in range(len(fileName))]
    y2 = [[] for _ in range(len(fileName))]

    for idx,filename in enumerate(fileName):

        with open(file_path+'{}.txt'.format(filename+'_'+par), 'r') as file:
            for line in file:
                # 如果文件中每行是以空格或制表符分隔的两列数据
                columns = line.split()  # 使用空格分隔数据
                if len(columns) >= 2:
                    y1[idx].append(float(columns[0]))
                    y2[idx].append(float(columns[1]))

    
    # 创建对应的x坐标
    x_data = list(range(1, 61))  # 生成与y数据长度相同的序列作为x坐标

    # 创建图形和坐标轴对象
    #plt.figure(figsize=(16, 6))
    fig,ax = plt.subplots(1,2,figsize=(12, 5))
    plt.rcParams.update({'font.size': 16}) 
    # 绘制折线
    for i,y_data in enumerate(y2):
        ax[0].plot(x_data, y_data[:60], label=labelName[i])
        #ax[0].set_xlabel('联邦学习聚合轮数',fontsize=16)
        #ax[0].set_ylabel('模型测试准确率', fontsize=16)
        ax[0].set_xlabel('Aggregation Round',fontsize=16)
        ax[0].set_ylabel('Test Accuracy(%)', fontsize=16)
        ax[0].set_ylim(50,100)
        ax[0].legend()
    

    for i,y_data in enumerate(y1):
        ax[1].plot(x_data, y_data[:60], label=labelName[i])
        # 添加标题和标签
        #ax[1].set_xlabel('联邦学习聚合轮数', fontsize=16)
        #ax[1].set_ylabel('模型测试损失值(log)', fontsize=16)
        ax[1].set_xlabel('Aggregation Round', fontsize=16)
        ax[1].set_ylabel('Test Loss', fontsize=16)
        #ax[1].set_yscale('log')

        ax[1].legend()
        # 显示图形
    
    plt.show()

def Draw_CNN2_FashionandCIFAR():
    file_path = 'D:/Graduation project/Code/FlowerBNN/results/FashionCNN2/'
    fileName1 = ['CNN2FF','CNN2BF','CNN2FB','CNN2BB']
    fileName2 = ['CIFAR'+name for name in fileName1]
    labelName = ['FF','BF','FB','BB']
    y1 = [[] for _ in range(len(fileName1))]
    y2 = [[] for _ in range(len(fileName2))]
    for idx,filename in enumerate(fileName1):
        with open(file_path+'{}.txt'.format(filename), 'r') as file:
            for line in file:
                # 如果文件中每行是以空格或制表符分隔的两列数据
                columns = line.split()  # 使用空格分隔数据
                if len(columns) >= 2:
                    y1[idx].append(float(columns[1]))


    for idx,filename in enumerate(fileName2):
        with open(file_path+'{}.txt'.format(filename), 'r') as file:
            for line in file:
                # 如果文件中每行是以空格或制表符分隔的两列数据
                columns = line.split()  # 使用空格分隔数据
                if len(columns) >= 2:
                    y2[idx].append(float(columns[1]))
    
    x_data = list(range(1, 44))  # 生成与y数据长度相同的序列作为x坐标

    # 创建图形和坐标轴对象
    #plt.figure(figsize=(16, 6))
    fig,ax = plt.subplots(1,2,figsize=(12, 5))
    plt.rcParams.update({'font.size': 16}) 
    # 绘制折线
    for i,y_data in enumerate(y1):
        ax[0].plot(x_data, y_data[:43], label=labelName[i])
        #ax[0].set_xlabel('联邦学习聚合轮数',fontsize=16)
        #ax[0].set_ylabel('模型测试准确率', fontsize=16)
        ax[0].set_xlabel('Aggregation Round',fontsize=16)
        ax[0].set_ylabel('Test Accuracy(%)', fontsize=16)
        ax[0].set_ylim(0,100)
        ax[0].legend()


    for i,y_data in enumerate(y2):
        ax[1].plot(x_data, y_data[:43], label=labelName[i])
        # 添加标题和标签
        #ax[1].set_xlabel('联邦学习聚合轮数', fontsize=16)
        #ax[1].set_ylabel('模型测试损失值(log)', fontsize=16)
        ax[1].set_xlabel('Aggregation Round', fontsize=16)
        ax[1].set_ylabel('Test Accuracy(%)', fontsize=16)
        ax[1].set_ylim(0,100)

        ax[1].legend()
        # 显示图形
    
    plt.show()
    
    

def Draw_Fashion_CNN2():
    #plt.rcParams['font.family'] = ['KaiTi','SimSun','Microsoft YaHei', 'SimHei']
    file_path = 'D:/Graduation project/Code/FlowerBNN/results/FashionCNN2/'

    # 读取文件并提取第二列数据作为y坐标
    fileName = ['CNN2FF','CNN2BF','CNN2FB','CNN2BB']
    labelName = ['FF','BF','FB','BB']
    y1 = [[] for _ in range(len(fileName))]
    y2 = [[] for _ in range(len(fileName))]

    for idx,filename in enumerate(fileName):

        with open(file_path+'{}.txt'.format(filename), 'r') as file:
            for line in file:
                # 如果文件中每行是以空格或制表符分隔的两列数据
                columns = line.split()  # 使用空格分隔数据
                if len(columns) >= 2:
                    y1[idx].append(float(columns[0]))
                    y2[idx].append(float(columns[1]))

    
    # 创建对应的x坐标
    x_data = list(range(1, 44))  # 生成与y数据长度相同的序列作为x坐标

    # 创建图形和坐标轴对象
    #plt.figure(figsize=(16, 6))
    fig,ax = plt.subplots(1,2,figsize=(12, 5))
    plt.rcParams.update({'font.size': 16}) 
    # 绘制折线
    for i,y_data in enumerate(y2):
        ax[0].plot(x_data, y_data[:43], label=labelName[i])
        #ax[0].set_xlabel('联邦学习聚合轮数',fontsize=16)
        #ax[0].set_ylabel('模型测试准确率', fontsize=16)
        ax[0].set_xlabel('Aggregation Round',fontsize=16)
        ax[0].set_ylabel('Test Accuracy(%)', fontsize=16)
        ax[0].set_ylim(0,100)
        ax[0].legend()
    

    for i,y_data in enumerate(y1):
        ax[1].plot(x_data, y_data[:43], label=labelName[i])
        # 添加标题和标签
        #ax[1].set_xlabel('联邦学习聚合轮数', fontsize=16)
        #ax[1].set_ylabel('模型测试损失值(log)', fontsize=16)
        ax[1].set_xlabel('Aggregation Round', fontsize=16)
        ax[1].set_ylabel('Test Loss(log)', fontsize=16)
        ax[1].set_yscale('log')

        ax[1].legend()
        # 显示图形
    
    plt.show()

def time_bar():
    plt.rcParams.update({'font.size': 16}) 
    Fmodel = [4.12, 106.51, 277.78]
    Bmodel = [0.08, 0.21, 0.19]
    x = np.arange(len(Fmodel))
    width = 0.4
    plt.figure(figsize=(12, 6))
    # 创建第一个子图
    plt.subplot(1, 2, 1)
    #plt.bar(x, Fmodel, color='lightskyblue', label='全加密聚合系统', width=width)
    #plt.bar(x , Bmodel, color='aquamarine', label='混合加密聚合系统', width=width)
    plt.bar(x, Fmodel, color='lightskyblue', label='Fully Encryption System', width=width)
    plt.bar(x , Bmodel, color='aquamarine', label='Partial Encryption System', width=width)
    plt.xticks(x , labels=['MLP', 'CNN2', 'CNN1'], fontsize=16)
    #plt.ylabel('密钥生成耗时(单位：秒(log))', fontsize=16)
    #plt.xlabel('网络结构', fontsize=16)
    plt.ylabel('Time (s(log))', fontsize=16)
    plt.xlabel('Model Infrastructure', fontsize=16)
    plt.yscale('log')
    plt.legend()

    # 创建第二个子图
    Fmodelexc = [13.21,277.46,610.38]
    Bmodelexc = [6.48,56.28,111.49]
    plt.subplot(1, 2, 2)
    #plt.bar(x, Fmodelexc, color='lightskyblue', label='全加密聚合系统', width=width)
    #plt.bar(x , Bmodelexc, color='aquamarine', label='混合加密聚合系统', width=width)
    plt.bar(x, Fmodelexc, color='lightskyblue', label='Fully Encryption System', width=width)
    plt.bar(x , Bmodelexc, color='aquamarine', label='Partial Encryption System', width=width)
    plt.xticks(x , labels=['MLP', 'CNN2', 'CNN1'], fontsize=16)
    #plt.ylabel('每轮参数聚合耗时(单位：秒)', fontsize=16)
    #plt.xlabel('网络结构', fontsize=16)
    plt.ylabel('Time(s))', fontsize=16)
    plt.xlabel('Model Infrastructure', fontsize=16)
    #plt.yscale('log')
    plt.legend()

    # 调整子图之间的间距
    plt.tight_layout()

    # 显示图形
    plt.show()

def Secure_MNIST_MLP():
    #plt.rcParams['font.family'] = ['KaiTi','SimSun','Microsoft YaHei', 'SimHei']
    file_path = 'D:/Graduation project/Code/FlowerBNN/results/SecAcc/'

    # 读取文件并提取第二列数据作为y坐标
    fileName = ['FMLP','BMLP','SecureFMLP','SecureBMLP']
    Name = ['MPL','BMLP','Sec_MLP','Sec_BMLP']
    y1 = [[] for _ in range(len(fileName))]
    y2 = [[] for _ in range(len(fileName))]

    for idx,filename in enumerate(fileName):

        with open(file_path+'{}.txt'.format(filename), 'r') as file:
            for line in file:
                # 如果文件中每行是以空格或制表符分隔的两列数据
                columns = line.split()  # 使用空格分隔数据
                if len(columns) >= 2:
                    y1[idx].append(float(columns[0]))
                    y2[idx].append(float(columns[1]))

    
    # 创建对应的x坐标
    x_data = list(range(1, 46))  # 生成与y数据长度相同的序列作为x坐标

    # 创建图形和坐标轴对象
    #plt.figure(figsize=(16, 6))
    fig,ax = plt.subplots(1,2,figsize=(12, 5))
    plt.rcParams.update({'font.size': 16}) 
    # 绘制折线
    for i,y_data in enumerate(y2):
        ax[0].plot(x_data, y_data[:45], label=Name[i])
        #ax[0].set_xlabel('联邦学习聚合轮数',fontsize=16)
        #ax[0].set_ylabel('模型测试准确率', fontsize=16)
        ax[0].set_xlabel('Aggregation Round',fontsize=16)
        ax[0].set_ylabel('Test Accuracy(%)', fontsize=16)
        ax[0].set_ylim(0,100)
        ax[0].legend()
    

    for i,y_data in enumerate(y1):
        ax[1].plot(x_data, y_data[:45], label=Name[i])
        # 添加标题和标签
        #ax[1].set_xlabel('联邦学习聚合轮数', fontsize=16)
        #ax[1].set_ylabel('模型测试损失值(log)', fontsize=16)
        ax[1].set_xlabel('Aggregation Round', fontsize=16)
        ax[1].set_ylabel('Test Loss(log)', fontsize=16)
        ax[1].set_yscale('log')

        ax[1].legend()
        # 显示图形
    
    plt.show()
if __name__ == '__main__':
    #Draw_Plot()
    #Cmprate()
    #Draw_MNIST_MLPandCNN2()
    #Draw_Fashion_CNN2()
    time_bar()
    #Secure_MNIST_MLP()
    #Draw_CNN2_FashionandCIFAR()