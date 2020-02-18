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

class Config():
    training_dir = "/home/ghy/GHY/THz-security/sdsn/data/data/train/"
    val_dir = "/home/ghy/GHY/THz-security/sdsn/data/data/val/"
    testing_dir = "/home/ghy/GHY/THz-security/sdsn/data/data/test/"
    train_batch_size = 64
    train_number_epochs = 30
    
cfg={
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


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
        )#4608-->10

    def forward_once(self, x):
        output = self.features(x)
        # print(output.size())
        output = output.view(output.size(0), -1)
        # print(output.size())#64*4608
        output=self.classifer1(output)
        output = self.classifer2(output)
        # print(output.size())
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



one_train = np.array([])


# load model
model = torch.load("/home/ghy/GHY/THz-security/sdsn/work_dir/vgg/net_29.pth")
net = model

#########################测试########################




frame = pd.read_csv('/home/ghy/GHY/THz-security/sdsn/data/data/val.csv')
index = 0
i = 1
list_d = 0
transform=transforms.Compose([transforms.Resize((100, 100)),
                              transforms.ToTensor()])



root_dir = "/home/ghy/GHY/THz-security/sdsn/data/data/val/tmp"



while index < 1600:   # val data

    img0_name=frame.iloc[index, 0]
    c=int(img0_name.split('_')[-1].split('.')[0])
    img0_path=os.path.join(root_dir,img0_name)
    if c in (1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15):
        # print('正常')
        d = 16 - c
        # print('c:{}, d:{}'.format(c, d))
        if c < 10:
            img1_name = frame.iloc[index, 0].split('.')[0][:-1] + str(d) + '.jpg'
        else:
            img1_name = frame.iloc[index, 0].split('.')[0][:-2] + str(d) + '.jpg'
        img1_path = os.path.join(root_dir + '/' + img1_name)
        # print('第一图：{} \n第二图：{}'.format(img0_name, img1_name))
        img1 = Image.open(img1_path)
        # print('img1 size:{}\nimg1:{}\nimg1 mode:{}'.format(img1.size, type(img1), img1.mode))
        img1 = img1.transpose(Image.FLIP_LEFT_RIGHT)
        img0 = Image.open(img0_path)
    else:
        # print('特殊')
        gen = Image.open(img0_path)
        img1_name = img0_name
        # print('第一图：{} \n第二图：{}'.format(img0_name, img1_name))
        img0 = gen.crop((0, 0, 256, 660))

        img1 = gen.crop((256, 0, 512, 660))
        # print('img1 size:{}\nimg1:{}\nimg1 mode:{}'.format(img1.size,type(img1),img1.mode))
        img1 = img1.transpose(Image.FLIP_LEFT_RIGHT)
    img0=transform(img0)
    img1=transform(img1)
    # print('img1_tfd size:{}'.format(img1.size()))
    img0=img0.unsqueeze(0)
    img1=img1.unsqueeze(0)
    label = frame.iloc[index, 1]
    label=torch.from_numpy(np.array(label, dtype=np.float32))
    x0,x1,label=img0.cuda(),img1.cuda(),label.cuda()
    # print('x0 size:{}\nx1:{}'.format(type(x0),type(x1)))
    #net=net('VGG16').cuda
    output1, output2 = net(x0, x1)

    euclidean_distance = F.pairwise_distance(output1, output2)
    # print('euclidean_distance type:{}'.format(euclidean_distance))
    euclidean_distance=euclidean_distance.cpu().detach().numpy()
    # print('euclidean_distance type:{}'.format(type(euclidean_distance)))


    ### idn ###
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
        mean_d=list_d                                                        # a-idn
        print('mean_d:{}'.format(mean_d))
        list_d=0
        one_train = np.append(one_train, mean_d)


    index+=1
    # print('index:{}'.format(index))
    i += 1




np.savetxt('gmm.txt', one_train)    #save to file



from math import sqrt, log, exp, pi
from random import uniform

#For plotting
import matplotlib.pyplot as plt
#for matrix math
import numpy as np
#for normalization + probability density function computation
from scipy import stats
#for plotting
import seaborn as sns
import pandas as pd


class Gaussian:
    "Model univariate Gaussian"

    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma

    def pdf(self, datum):
        "Probability of a data point given the current parameters"

        u = (datum - self.mu) / abs(self.sigma)
        y = (1 / (sqrt(2 * pi) * abs(self.sigma))) * exp(-u * u / 2)
        return y

    def __repr__(self):

        return 'Gaussian({0:4.6}, {1:4.6})'.format(self.mu, self.sigma)

from random import uniform
from math import sqrt, log, exp, pi
data = np.loadtxt('/home/ghy/GHY/THz-security/sdsn/src/gmm.txt')
# print(data)


class GaussianMixture:
    "Model mixture of two univariate Gaussians and their EM estimation"

    def __init__(self, data, mu_min=min(data), mu_max=max(data), sigma_min=.1, sigma_max=1, mix=.5):
        self.data = data
        self.one = Gaussian(uniform(mu_min, mu_max),
                            uniform(sigma_min, sigma_max))
        self.two = Gaussian(uniform(mu_min, mu_max),
                            uniform(sigma_min, sigma_max))
        self.mix = mix

    def Estep(self):
        "Perform an E(stimation)-step, freshening up self.loglike in the process"
        # compute weights
        self.loglike = 0.  # = log(p = 1)
        for datum in self.data:
            # unnormalized weights
            wp1 = self.one.pdf(datum) * self.mix
            wp2 = self.two.pdf(datum) * (1. - self.mix)
            # compute denominator
            den = wp1 + wp2
            # normalize
            wp1 /= den
            wp2 /= den
            # add into loglike
            self.loglike += log(wp1 + wp2)
            # yield weight tuple
            yield (wp1, wp2)

    def Mstep(self, weights):
        "Perform an M(aximization)-step"
        # compute denominators
        (left, rigt) = zip(*weights)
        one_den = sum(left)
        two_den = sum(rigt)
        # compute new means
        self.one.mu = sum(w * d / one_den for (w, d) in zip(left, data))
        self.two.mu = sum(w * d / two_den for (w, d) in zip(rigt, data))
        # compute new sigmas
        self.one.sigma = sqrt(sum(w * ((d - self.one.mu) ** 2)
                                  for (w, d) in zip(left, data)) / one_den)
        self.two.sigma = sqrt(sum(w * ((d - self.two.mu) ** 2)
                                  for (w, d) in zip(rigt, data)) / two_den)
        # compute new mix
        self.mix = one_den / len(data)
        # print("one mu = {}, two mu = {}".format(self.one.mu, self.two.mu))
    def iterate(self, N=1, verbose=False):
        "Perform N iterations, then compute log-likelihood"
        for i in range(1, N + 1):
            self.Mstep(self.Estep())
            if verbose:
                print('{0:2} {1}'.format(i, self))
        self.Estep()  # to freshen up self.loglike

    def pdf(self, x):
        return (self.mix) * self.one.pdf(x) + (1 - self.mix) * self.two.pdf(x)

    def __repr__(self):
        return 'GaussianMixture({0}, {1}, mix={2.03})'.format(self.one,
                                                              self.two,
                                                              self.mix)

    def __str__(self):
        return 'Mixture: {0}, {1}, mix={2:.03})'.format(self.one,
                                                        self.two,
                                                        self.mix)


# Find best Mixture Gaussian model
n_iterations = 100
n_random_restarts = 50
best_mix = None
best_loglike = float('-inf')
print('Computing best model with random restarts...\n')
for _ in range(n_random_restarts):
    mix = GaussianMixture(data)
    for _ in range(n_iterations):
        try:
            mix.iterate()
            if mix.loglike > best_loglike:
                best_loglike = mix.loglike
                best_mix = mix
        except (ZeroDivisionError, ValueError, RuntimeWarning): # Catch division errors from bad starts, and just throw them out...
            pass
print('\n\nDone. 🙂')
mu1 = max(mix.one.mu, mix.two.mu)
mu2 = min(mix.one.mu, mix.two.mu)

x = np.linspace(-1, 15, 200)
#mixture
sns.distplot(data, bins=20, kde=False, norm_hist=True)
g_both = [best_mix.pdf(e) for e in x]
plt.subplot(2, 1, 1)
plt.plot(x, g_both, label='gaussian mixture')
plt.legend()
plt.savefig("gmm.png")

y = 0
a = 0
n = 0
b = 0
q = 0
g = 0


scores = np.array([])

intervals = np.linspace(mu2, mu1, 20)
for interval in intervals:
    i = 1
    list_d = 0
    index = 0
    while index < 1600:   # val data

        img0_name=frame.iloc[index, 0]
        c=int(img0_name.split('_')[-1].split('.')[0])
        img0_path=os.path.join(root_dir,img0_name)
        if c in (1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15):
            # print('正常')
            d = 16 - c
            # print('c:{}, d:{}'.format(c, d))
            if c < 10:
                img1_name = frame.iloc[index, 0].split('.')[0][:-1] + str(d) + '.jpg'
            else:
                img1_name = frame.iloc[index, 0].split('.')[0][:-2] + str(d) + '.jpg'
            img1_path = os.path.join(root_dir + '/' + img1_name)
            # print('第一图：{} \n第二图：{}'.format(img0_name, img1_name))
            img1 = Image.open(img1_path)
            # print('img1 size:{}\nimg1:{}\nimg1 mode:{}'.format(img1.size, type(img1), img1.mode))
            img1 = img1.transpose(Image.FLIP_LEFT_RIGHT)
            img0 = Image.open(img0_path)
        else:
            # print('特殊')
            gen = Image.open(img0_path)
            img1_name = img0_name
            # print('第一图：{} \n第二图：{}'.format(img0_name, img1_name))
            img0 = gen.crop((0, 0, 256, 660))

            img1 = gen.crop((256, 0, 512, 660))
            # print('img1 size:{}\nimg1:{}\nimg1 mode:{}'.format(img1.size,type(img1),img1.mode))
            img1 = img1.transpose(Image.FLIP_LEFT_RIGHT)
        img0=transform(img0)
        img1=transform(img1)
        # print('img1_tfd size:{}'.format(img1.size()))
        img0=img0.unsqueeze(0)
        img1=img1.unsqueeze(0)
        label = frame.iloc[index, 1]
        label=torch.from_numpy(np.array(label, dtype=np.float32))
        x0,x1,label=img0.cuda(),img1.cuda(),label.cuda()
        # print('x0 size:{}\nx1:{}'.format(type(x0),type(x1)))
        #net=net('VGG16').cuda
        output1, output2 = net(x0, x1)

        euclidean_distance = F.pairwise_distance(output1, output2)
        # print('euclidean_distance type:{}'.format(euclidean_distance))
        euclidean_distance=euclidean_distance.cpu().detach().numpy()
        # print('euclidean_distance type:{}'.format(type(euclidean_distance)))


        ### idn ###
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
                mean_d=list_d                                                       # a-idn
                print('mean_d:{}'.format(mean_d))
                list_d=0
                
                if (mean_d > interval):  # 大于1就是true，就是1
                    print('该人有异物,label:{}'.format(label))
                    y += 1  # 检测出有异物的总数
                    if (label == 1):

                        a += 1  # 有异物检测正确
                    else:

                        print('****************label=0却测出异物************************')
                        g+=1
                else:
                    print('该人没有异物,label:{}'.format(label))
                    n += 1  # 检测出无异物的总数
                    if (label == 0):
                        b += 1  # 无异物检测正确
                    else:
                        print('___________________label=1却测出无异物______________________')
                        q+=1
        index+=1
        # print('index:{}'.format(index))
        i += 1
    
    n_p = a / (y)   # negetive查准率
    n_r = a / (a+q)#查全率
    # print('有异物查准率:{} \n查全率:{}'.format(n_p, n_r))
    F1 = (2 * n_p * n_r) / (n_p + n_r)
    scores = np.append(scores, F1)
    print("scores =", scores)

plt.subplot(2, 1, 2)
plt.plot(intervals, scores, label="x: theta, y: F1")
plt.legend()
plt.savefig("F1.png")