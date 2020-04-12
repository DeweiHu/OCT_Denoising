# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 23:05:04 2020

@author: hudew
"""

import sys
sys.path.insert(0,'C:\\Users\\hudew\\OneDrive\\桌面\\Denoise\\')
import MyFunctions

import numpy as np
import pickle
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import StepLR
from torch.autograd import Variable

global root
root = 'E:\\Denoise_Train_Data\\'
modelroot = 'E:\\Model\\'

#%%
print('Creating dataset...')

batch_size = 1
gpu = 1

class MyDataset(Data.Dataset):

    def ToTensor(self, image, mask):
        x_tensor = torch.tensor(image).type(torch.FloatTensor)
        y_tensor = transforms.functional.to_tensor(mask)
        x_tensor = torch.unsqueeze(x_tensor,dim=0)
        return x_tensor, y_tensor    
        
    def __init__(self, dataroot):
        with open(dataroot,'rb') as f:
            self.pair = pickle.load(f)
        
    def __len__(self):
        return len(self.pair)

    def __getitem__(self,idx):
        (image, mask) = self.pair[idx]
        # only load the original noisy image channel
        x_tensor, y_tensor = self.ToTensor(image[0,:,:],
                                MyFunctions.ImageRescale(mask,[0,1]))
        return x_tensor, y_tensor
    
train_loader = Data.DataLoader(dataset=MyDataset(root+'Human_train.pickle'), 
                               batch_size=batch_size, shuffle=True)

test_loader = Data.DataLoader(dataset=MyDataset('E:\\Denoise_Test_Data\\Retina2_ONH_SNR_101_2.pickle'), 
                               batch_size=batch_size, shuffle=True)

device = torch.device("cuda:0" if( torch.cuda.is_available() and gpu>0 ) else "cpu")

#%% Generator Architecture
print('Initializing model...')

def weight_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def down_block(in_channels,out_channels):
    return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=4,
                      stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ELU()
            )
    
def up_block(in_channels,out_channels):
    return nn.Sequential(
            nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels,
                               kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ELU()
            )

def Standard_block(in_channels,out_channels):
     return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                      kernel_size=3, stride=1, padding=1),
            nn.ELU(),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels,
                      kernel_size=3, stride=1, padding=1)
            )

def double_path(in_channels,d):
    return nn.Sequential(
           nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=3,
                     stride=1, padding=d, dilation=d),
           nn.ELU(),
           nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3,
                     stride=1, padding=d, dilation=d),
           nn.BatchNorm2d(64),
           nn.ELU()
           )

def single_path(in_channels,d):
    return nn.Sequential(
           nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=3,
                     stride=1, padding=d, dilation=d),
           nn.ELU()
           )
    
def conv(in_channels,out_channels):
    return nn.Sequential(
           nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                     kernel_size=1),
           nn.ELU()
           )
    
class UNet(nn.Module):
    def __init__(self,gpu):
        super(UNet,self).__init__()
        self.gpu = gpu
        
        self.down = down_block(64,64)
        self.up = up_block(64,64)
        
        self.Standard_1 = Standard_block(1,64)
        
        self.Res1_double = double_path(64,2)
        self.Res1_single = single_path(64,2)
        
        self.Res2_double = double_path(64,4)
        self.Res2_single = single_path(64,4)

        self.Standard_2 = Standard_block(64,64)
        
        self.Res3_double = double_path(128,4)
        self.Res3_single = single_path(128,4)
        
        self.Standard_3 = Standard_block(128,64)
        
        self.conv = conv(64,64)
        self.convopt = conv(256,1)
        
    def forward(self,x):
        # Downwards
        layer_1 = self.Standard_1(x)           #[64,1024,512]
        x = self.down(layer_1)                 #[64,512,256]
        
        layer_2 = torch.add(self.Res1_double(x),self.Res1_single(x))  #[64,512,256]
        x = self.down(layer_2)                                        #[64,256,128]

        layer_3 = torch.add(self.Res2_double(x),self.Res2_single(x))  #[64,256,128]
        x = self.down(layer_3)                                        #[64,128,64]
        
        x = self.Standard_2(x)
        
        # Upwards
        x = self.up(x)                                          #[64,256,128]
        x = torch.cat([x,layer_3],dim=1)                        #[128,256,128]
        x = torch.add(self.Res3_double(x),self.Res3_single(x))  #[64,256,128]

        x = self.up(x)                                          #[64,512,256]
        x = torch.cat([x,layer_2],dim=1)                        #[128,512,256]
        x = torch.add(self.Res3_double(x),self.Res3_single(x))  #[64,512,256]
        
        x = self.up(x)                         #[64,1024,512]
        x = torch.cat([x,layer_1],dim=1)       #[128,1024,512]
        x = self.Standard_3(x)                 #[64,1024,512]
        
        # Middle feedbacks
        l1 = self.conv(layer_1)
        l2 = self.up(self.conv(layer_2))
        l3 = self.up(self.up(self.conv(layer_3)))
        x = torch.cat([x,l1],dim=1)
        x = torch.cat([x,l2],dim=1)
        x = torch.cat([x,l3],dim=1)
        
        output = self.convopt(x)
        return output
        
model = UNet(gpu).to(device)
#model.apply(weight_init)
#name = 'Multiscale_UNet_human.pt'
#model.load_state_dict(torch.load(modelroot+name))

#%% Loss functions and optimizers
L1_loss = nn.L1Loss()

beta1 = 0.5
lr = 0.0001
optimizer = optim.Adam(model.parameters(),lr=lr,betas=(beta1,0.999))
scheduler = StepLR(optimizer, step_size=1, gamma=0.5)

#%%
print('Training start...')

import time

num_epoch = 2

t1 = time.time()
for epoch in range(num_epoch):
    for step,[x,y] in enumerate(train_loader):
        
#        model.train()
        
        train_x = Variable(x).to(device)        
        train_y = Variable(y).to(device)
        pred = model(train_x)
        
        loss = L1_loss(pred,train_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if step % 100 == 0:
            print('[%d/%d][%d/%d]\tLoss:%.4f'
                  %(epoch, num_epoch, step, len(train_loader),loss))
            
        if step % 1000 == 0:
            noise_img = train_x.detach().cpu().numpy()
            denoise_img = pred.detach().cpu().numpy()
            average_img = train_y.detach().cpu().numpy()
            
            plt.figure(figsize=(12,12))
            plt.subplot(1,3,1),plt.axis('off'),
            plt.imshow(noise_img[0,0,:,:500],cmap='gray'),plt.title('noisy')
            plt.subplot(1,3,2),plt.axis('off'),
            plt.imshow(denoise_img[0,0,:,:500],cmap='gray'),plt.title('denoised')
            plt.subplot(1,3,3),plt.axis('off'),
            plt.imshow(average_img[0,0,:,:500],cmap='gray'),plt.title('5-average')
            plt.show()
    
    scheduler.step()
    print('//////////////////////////////////////////////////')
    print('test...')
#    model.eval()
    with torch.no_grad():
        for slc,[x,y] in enumerate(test_loader):
            
            if slc > 5 :
                break

            test_x = Variable(x).to(device)
            pred_y = model(test_x)
            
            img_noi = x.numpy()
            img_gt = y.numpy()
            img_dn = pred_y.detach().cpu().numpy()
            
            plt.figure(figsize=(12,12))
            plt.subplot(1,3,1),plt.axis('off'),
            plt.imshow(img_noi[0,0,:,:500],cmap='gray'),plt.title('noisy')
            plt.subplot(1,3,2),plt.axis('off'),
            plt.imshow(img_dn[0,0,:,:500],cmap='gray'),plt.title('denoised')
            plt.subplot(1,3,3),plt.axis('off'),
            plt.imshow(img_gt[0,0,:,:500],cmap='gray'),plt.title('5-average')
            plt.show()
            
#%% Save model as GPU version
modelroot = 'E:\\Model\\'
name = 'Multiscale_UNet_human.pt'
torch.save(model.state_dict(), modelroot+name)
