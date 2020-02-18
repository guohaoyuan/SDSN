import csv
from torch import optim
import torch.nn as nn
import os
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader,Dataset
import torch 
from torchvision import transforms 
from torchvision.datasets import ImageFolder
import random
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt
import PIL


cfg={
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

def save_network(net, epoch_number): #
    save_filename = 'net_%s.pth'% epoch_number
    save_path = os.path.join('/home/ghy/GHY/THz-security/sdsn/work_dir/vgg', save_filename)
    torch.save(net, save_path)



def imshow(img,text=None,should_save=False):
    npimg = img.numpy()
    plt.axis("off")
    if text:
        plt.text(75, 8, text, style='italic',fontweight='bold',
            bbox={'facecolor':'white', 'alpha':0.8, 'pad':10})
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()    

def show_plot(iteration, epoch_number,loss):
    fig = plt.figure()
    plt.plot(iteration, loss)
    #plt.show()
    name = 'train_epoches_%d.jpg' % epoch_number
    fig.savefig(os.path.join('/home/ghy/GHY/THz-security/sdsn/work_dir/vgg', name))


class Config():
    training_dir = "/home/ghy/GHY/THz-security/sdsn/data/data/train/"
    val_dir = "/home/ghy/GHY/THz-security/sdsn/data/data/val/"
    testing_dir = "/home/ghy/GHY/THz-security/sdsn/data/data/test/"
    train_batch_size = 64
    train_number_epochs = 30

    
class SiameseNetworkDataset(Dataset):
    def __init__(self,imageFolderDataset,train=True,transform=None,should_invert=True):
        self.imageFolderDataset = imageFolderDataset
        self.train = train
        self.transform = transform
        self.should_invert = should_invert

    def __getitem__(self,index):
        
        
        if self.train:              # train
            i = random.randint(0,15)  # random sample
            q = random.randint(0,1)   # random sample
            if q:
                # print('负样本')      # negative
                list2 = os.listdir('/home/ghy/GHY/THz-security/sdsn/data/data/train/0')         # negative sample path
                j = random.sample(list2, 1)                                                            # random sample one id
                # print("join befor j = {}".format(j))
                j = ','.join(j)             # list 变成字符串
                # print("join after j = {}".format(j))
                #imgs = os.listdir('C:/Users/lenovo/Desktop/an/folder/01')
                path = os.path.join('/home/ghy/GHY/THz-security/sdsn/data/data/train/0'+'/'+j)#第j个文件夹
                
                #print(path)
                imgs = os.listdir(path)
                #print('imgs的list:{}'.format(imgs))
                imgs = sorted(imgs,key=lambda x:int(x.split('/')[-1].split('_')[-1].split('.')[0]))
                #print('imgs排序后:{}'.format(imgs))
                #print('文件夹地址：{}'.format(path))
                img0 = os.path.join(path+'/'+imgs[i])                                             # 与下面img0的等价
                #a=img0.split('/')[-1].split('_')[0]#为了下面的label判断语句
                # print('图1:{}'.format(img0))
                #img0 = Image.open(img0)#open
                #img0=os.path.join('C:/Users/lenovo/Desktop/an/folder/01',imgs[i])#是一个路径
                
                c = img0.split('/')[-1].split('_')[-1].split('.')[0]                              # 得到img0的序号
                c = int(c)
                #print('图1的标号：{}'.format(c))
                #img0.show()
                if c in (1,2,3,4,5,6,7,9,10,11,12,13,14,15):                                      # 一个id有16个图片，此处不选择0与180两个角度的图片
                    #print('正常')
                    d = 16 - c                                                                    # 得到角度对称的另一张图片
                    #print('图2的序号：{}'.format(16-c))
                    #print('imgs排序后2:{}'.format(imgs))
                    #print('文件夹的地址2：{}'.format(path))
                    #print('图片2名字：{}'.format(imgs[d]))
                    img1name = os.path.join(path+'/'+imgs[d])
                    # print('第二图：{}'.format(img1name))
                    img1 = Image.open(path+'/'+imgs[d])#imgs是随机取出来的
                    img1 = img1.transpose(Image.FLIP_LEFT_RIGHT)                                     # img1水平翻转和img0形成一个image pair
                                    
                    img0 = Image.open(img0)
                    #print('对应的图2：{}'.format(img1name))
                    #img0.show()
                    #img1.show()
                else:               #正反面
                    #print('特殊')
                    gen = Image.open(img0)
                    img1name = img0
                    # print('第二图:{}'.format(img1name))
                    img0 = gen.crop((0,0,256,660))
                    img1 = gen.crop((256,0,512,660))
                    img1 = img1.transpose(Image.FLIP_LEFT_RIGHT)                                     # img0中心轴线裁开，水平翻转得到img1与img0形成嗯image pair
                    #img0.show()
                    #img1.show()

                label = 0             # label为0
            else:                     # 正样本
                # print('正样本')
                i = random.randint(0,15)
                list2 = os.listdir('/home/ghy/GHY/THz-security/sdsn/data/data/train/1')
                j = random.sample(list2,1)
                j = ','.join(j) 
                path = os.path.join('/home/ghy/GHY/THz-security/sdsn/data/data/train/1'+'/'+j)
                
                #print(path)
                imgs = os.listdir(path)
                imgs = sorted(imgs,key=lambda x:int(x.split('/')[-1].split('_')[-1].split('.')[0]))
                #print(imgs)
                img0 = os.path.join(path+'/'+imgs[i])#与下面img0的等价
                a = img0.split('/')[-1].split('_')[0]#为了下面的label判断语句
                #print(img0)
                #img0 = Image.open(img0)#open
                #img0=os.path.join('C:/Users/lenovo/Desktop/an/folder/01',imgs[i])#是一个路径
                # print('第一图:{}'.format(img0))
                c = img0.split('/')[-1].split('_')[-1].split('.')[0]
                c = int(c)
                #print(c)
                #img0.show()
                if c in (1,2,3,4,5,6,7,9,10,11,12,13,14,15):
                    #print('正常')
                    d = 16-c
                    #print(16-c)
                    img1 = Image.open(path+'/'+imgs[d])#imgs是随机取出来的，不能这样做
      
                    img1name = os.path.join(path+'/'+imgs[d])
                    # print('第二图:{}'.format(img1name))
                    img1 = img1.transpose(Image.FLIP_LEFT_RIGHT)
                    img0 = Image.open(img0)
                    #print(img1name)
                    #img0.show()
                    #img1.show()
                else:               #正反面
                    #print('特殊')
                    gen = Image.open(img0)
                    img1name = img0
                    # print('第二图：{}'.format(img1name))
                    img0 = gen.crop((0,0,256,660))
                    img1 = gen.crop((256,0,512,660))
                    img1 = img1.transpose(Image.FLIP_LEFT_RIGHT)
                    #img0.show()
                    #img1.show()

                label=1





        
        if self.should_invert:
            img0 = PIL.ImageOps.invert(img0)
            img1 = PIL.ImageOps.invert(img1)
            
        if self.transform is not None:
            img0 = self.transform(img0)
            img1 = self.transform(img1)

        return img0,img1,torch.from_numpy(np.array(label,dtype=np.float32))
    
    def __len__(self):
        return len(self.imageFolderDataset.imgs)
    




folder_dataset=ImageFolder(root=Config.training_dir)
siamese_dataset = SiameseNetworkDataset(imageFolderDataset=folder_dataset,
                                        transform=transforms.Compose([transforms.Resize((100,100)),
                                                                      transforms.ToTensor()
                                                                      ])
                                       ,should_invert=False)

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
        loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))


        return loss_contrastive


###############训练################


train_dataloader = DataLoader(siamese_dataset,
                        shuffle=True,
                        num_workers=8,
                        batch_size=Config.train_batch_size)


net = Sia_VGG('VGG16').cuda()

criterion = ContrastiveLoss()
optimizer = optim.Adam(net.parameters(),lr = 0.0005)
counter = []
loss_history = [] 
iteration_number= 0


for epoch in range(0,Config.train_number_epochs):
    train_acc=0
    cot=0
    for i, data in enumerate(train_dataloader,0):
        img0, img1, label = data

        img0, img1, label = img0.cuda(), img1.cuda() , label.cuda()
        # print('两张图的大小：{}{}'.format(img0.size(),img1.size()))
        optimizer.zero_grad()
        output1,output2 = net(img0,img1)
        #print('output1的大小为：{}'.format(output1.size()))
        loss_contrastive = criterion(output1,output2,label)
        D=torch.mean(F.pairwise_distance(output1, output2,p=2))

        # print('D大小为:{}'.format(D))
        y_pred=D>0.05#阈值为0.4
        # print('预测值:{}'.format(y_pred))
        train_acc+=float(torch.sum(y_pred==0))#以后需要修改
        cot+=1
        loss_contrastive.backward()
        optimizer.step()
        #print('取余数：{}'.format(i%10))
        # print('i的值：{}'.format(i))
        if i % 10 == 0:
            print("Epoch number {}\n Current loss {}\n".format(epoch, loss_contrastive.item()))
            iteration_number += 10
            counter.append(iteration_number)
            loss_history.append(loss_contrastive.item())

    train_acc = train_acc / cot
    print('每个epoch的acc为：{}'.format(train_acc))

    if epoch % 5 == 4:
        show_plot(counter, epoch, loss_history)

        save_network(net, epoch)

show_plot(counter, epoch, loss_history)

save_network(net, epoch)


