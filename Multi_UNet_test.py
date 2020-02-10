#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 12:25:02 2020

@author: hud4
"""

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
import torchvision.datasets as Dataset
import torchvision.transforms as transforms
from torch.autograd import Variable
import os

root = '/home/hud4/Desktop/2020/Human/'
filelist_x = []
filelist_y = []

for file in os.listdir(root): 
    if file.endswith("_x.npy"):
        filelist_x.append(file)
    if file.endswith("_y.npy"):
        filelist_y.append(file)

filelist_x.sort()
filelist_y.sort()

file_x = filelist_x[5]
file_y = filelist_y[5]

#%%
print('Creating dataset...')

batch_size = 1
gpu = 1

class MyDataset(Data.Dataset):
    
    def ToTensor(self, image, mask):
        x_tensor = torch.tensor(image).type(torch.FloatTensor)
        x_tensor = torch.unsqueeze(x_tensor,dim=0)
        y_tensor = transforms.functional.to_tensor(mask)
        return x_tensor, y_tensor
    
    def __init__(self,x_dir,y_dir):
        self.pair = ()
        self.train_data = np.load(x_dir)
        self.train_label = np.load(y_dir)
        self.num = self.train_data.shape[0]
        
        for i in range(self.num):
            x = self.train_data[i,:,:]
            y = self.train_label[i,:,:]
            self.pair = self.pair+((x,y),)
            
    def __len__(self):
        return self.num
    
    def __getitem__(self,idx):
        (image, mask) = self.pair[idx]
        x_tensor, y_tensor = self.ToTensor(image, mask)
        return x_tensor, y_tensor

test_loader = Data.DataLoader(dataset=MyDataset(root+file_x,root+file_y),
                               batch_size=batch_size, shuffle=False)

device = torch.device("cuda:0" if(torch.cuda.is_available() and gpu>0) else "cpu")

#%% Generator Architecture
print('Initializing model...')

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

#%% Load the model
modelroot = '/home/hud4/Desktop/2020/Model/'
G_name = 'Multi-UNet.pt'

model.load_state_dict(torch.load(modelroot+G_name))

shape = [500,1024,500]
denoise_volume = np.zeros(shape).astype(np.float32)
average_volume = np.zeros(shape).astype(np.float32)
noisy_volume = np.zeros(shape).astype(np.float32)

#%% 
def cw90(img):
    [r,c] = img.shape
    opt = np.zeros([c,r])
    for i in range(r):
        vector = np.transpose(img[i,:])
        opt[:,r-i-1] = vector
    return opt

with torch.no_grad():
    for step,(test_x,test_y) in enumerate(test_loader):
        x = Variable(test_x).to(device)
        
        denoise = model(x).detach().cpu().numpy()
        average = test_y.numpy()
        noisy = test_x.detach().cpu().numpy()
        
        denoise_volume[:,:,step] = cw90(denoise[0,0,:,:500])
        average_volume[:,:,step] = cw90(average[0,0,:,:500])
        noisy_volume[:,:,step] = cw90(noisy[0,0,:,:500])
        
        if step % 50 == 0:
            plt.figure(figsize=(15,12))
            plt.subplot(1,3,1),plt.imshow(noisy[0,0,:,:500],cmap='gray')
            plt.subplot(1,3,2),plt.imshow(denoise[0,0,:,:500],cmap='gray')
            plt.subplot(1,3,3),plt.imshow(average[0,0,:,:500],cmap='gray')
            plt.show()
        
#%% Save volume 
def save_nii(volume,path,filename):
    output = nib.Nifti1Image(volume,np.eye(4))
    nib.save(output,os.path.join(path,filename))

name = file_x[:-10]
save_nii(denoise_volume,'/home/hud4/Desktop/','denoise_{}.nii.gz'.format(name))
save_nii(average_volume,'/home/hud4/Desktop/','average_{}.nii.gz'.format(name))
save_nii(noisy_volume,'/home/hud4/Desktop/','noisy_{}.nii.gz'.format(name))


