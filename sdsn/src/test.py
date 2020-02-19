#-*-coding:utf-8-*-
import csv
from torch import optim
import torch.nn as nn
import os
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder
import random
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt
import pandas as pd
# from scipy.interpolate import spline


cfg={
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class Config():
    training_dir = "/home/ghy/GHY/THz-security/sdsn/data/data/train/"
    val_dir = "/home/ghy/GHY/THz-security/sdsn/data/data/val/"
    testing_dir = "/home/ghy/GHY/THz-security/sdsn/data/data/test/"
    train_batch_size = 64
    train_number_epochs = 30
    


##################孪生网络###############
class Sia_VGG(nn.Module):
    #初始化参数
    def __init__(self,vgg_name):
        super(Sia_VGG, self).__init__()
        self.features=self._make_layers(cfg[vgg_name])#从字典中选出对应的序列赋值
        self.classifer1=nn.Linear(4608, 100)
        self.classifer2=nn.Sequential(
            nn.ReLU(True),
            nn.Linear(100, 5),
        )

    def forward_once(self, x):
        output = self.features(x)
        
        output = output.view(output.size(0), -1)
        
        output=self.classifer1(output)
        output = self.classifer2(output)
        
        return output

    #前向传播
    def forward(self,input1,input2):
        output1=self.forward_once(input1)
        output2=self.forward_once(input2)

        return output1,output2

    def _make_layers(self,cfg):
        layers=[]
        in_channels=1
        for x in cfg:
            if x=='M':
                layers+=[nn.MaxPool2d(kernel_size=2,stride=2)]
            else:
                layers+=[nn.Conv2d(in_channels,x,kernel_size=3,padding=1),
                         nn.BatchNorm2d(x),
                         nn.ReLU(inplace=True)]
                in_channels=x
        layers+=[nn.AvgPool2d(kernel_size=1,stride=1)]
        
        return nn.Sequential(*layers)


###############对比损失################
class ContrastiveLoss(torch.nn.Module):

    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))

        return loss_contrastive


    




# load model
model = torch.load("/home/ghy/GHY/THz-security/sdsn/work_dir/vgg/net_29.pth")

net = model

#########################测试########################

y = 0
a = 0
n = 0
b = 0
q = 0
g = 0


frame = pd.read_csv('/home/ghy/GHY/THz-security/sdsn/data/data/test.csv')
index = 0
i = 1
list_d = 0
transform=transforms.Compose([transforms.Resize((100, 100)),
                              transforms.ToTensor()])



root_dir = "/home/ghy/GHY/THz-security/sdsn/data/data/test"



while index < 3200:   # test data

    img0_name=frame.iloc[index, 0]
    c=int(img0_name.split('_')[-1].split('.')[0])
    img0_path=os.path.join(root_dir,img0_name)
    if c in (1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15):
        
        d = 16 - c
       
        if c < 10:
            img1_name = frame.iloc[index, 0].split('.')[0][:-1] + str(d) + '.jpg'
        else:
            img1_name = frame.iloc[index, 0].split('.')[0][:-2] + str(d) + '.jpg'
        img1_path = os.path.join(root_dir + '/' + img1_name)
        
        img1 = Image.open(img1_path)
        
        img1 = img1.transpose(Image.FLIP_LEFT_RIGHT)
        img0 = Image.open(img0_path)
    else:
        
        gen = Image.open(img0_path)
        img1_name = img0_name
        
        img0 = gen.crop((0, 0, 256, 660))

        img1 = gen.crop((256, 0, 512, 660))
        
        img1 = img1.transpose(Image.FLIP_LEFT_RIGHT)
    img0=transform(img0)
    img1=transform(img1)
    
    img0=img0.unsqueeze(0)
    img1=img1.unsqueeze(0)
    label = frame.iloc[index, 1]
    label=torch.from_numpy(np.array(label, dtype=np.float32))
    x0,x1,label=img0.cuda(),img1.cuda(),label.cuda()
    
    output1, output2 = net(x0, x1)

    euclidean_distance = F.pairwise_distance(output1, output2)
    
    euclidean_distance=euclidean_distance.cpu().detach().numpy()

    ##A idn ###
    if c == 0:
        list_d += 0.09 * euclidean_distance
    elif c == 1 or c == 15:
        list_d += 0.045 * euclidean_distance
    elif c == 2 or c == 14:
        list_d += 0.06 * euclidean_distance
    elif c == 3 or c == 13:
        list_d += 0.06 * euclidean_distance
    elif c == 4 or c == 12:
        list_d += 0.055 * euclidean_distance
    elif c == 5 or c == 11:
        list_d += 0.055 * euclidean_distance
    elif c == 6 or c == 10:
        list_d += 0.065 * euclidean_distance
    elif c == 7 or c == 9:
        list_d += 0.065 * euclidean_distance
    elif c == 8:
        list_d += 0.1 * euclidean_distance


    if i%16==0:
        mean_d=list_d                                                      
        
        list_d=0
        if (mean_d > 2.5):  # best threshold, we select the best threshold with the best F1 according to  F1.png. 
            # print('该人有异物,label:{}'.format(label))
            y += 1  # 检测出有异物的总数
            if (label == 1):

                a += 1  # 有异物检测正确
            else:

                print('****************label=0却测出异物************************')
                print("img name =", img0_name)
                g+=1
        else:
            # print('该人没有异物,label:{}'.format(label))
            n += 1  # 检测出无异物的总数
            if (label == 0):
                b += 1  # 无异物检测正确
            else:
                print('___________________label=1却测出无异物______________________')
                print("img name =", img0_name)
                q+=1
    index+=1
 
    i += 1
print('y:{}\na:{}\nb:{}\nq:{}\ng:{}'.format(y,a,b,q,g))
n_p = a / y  # negetive查准率
n_r = a / (a+q)#查全率
print('有异物查准率:{} \n查全率:{}'.format(n_p, n_r))


