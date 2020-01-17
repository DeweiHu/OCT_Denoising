#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 13:35:15 2020

@author: hud4
"""

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from medpy.filter.smoothing import anisotropic_diffusion
import cv2

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
import torchvision.datasets as Dataset
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import StepLR
from torch.autograd import Variable

print('Loading data...')

global dataroot, file_x, file_y
dataroot = '/home/hud4/Desktop/2020/Data/'
file_x = '2019-09-18-001_OD_V_6x6_0_0000003_structure.nii'
file_y = '2019-09-18-001_OD_V_6x6_0_0000003_structure_5avg.nii'

#%%
print('Creating dataset...')

global n_channel,im_size
batch_size = 1
n_channel = 5
im_size = 512
gpu = 1

class Test_Dataset(Data.Dataset):
    
    def Center_Crop(self, data, im_size):
        [r,c,n] = data.shape
        data_crop = np.zeros([im_size,im_size,n]).astype(np.uint8)
        left = np.int((c-im_size)/2)
        right = c-left
        top = np.int((r-im_size)/2)
        bottom = r-top
        for i in range(n):
            img = data[:,:,i]
            data_crop[:,:,i] = img[top:bottom,left:right]
        return data_crop

    def Grad_map(self, img, niter, kernel_size):
        diffuse = anisotropic_diffusion(img,niter=niter,option=2)
        diffuse = np.uint8(diffuse)
        sobelx = cv2.Sobel(diffuse,cv2.CV_64F,1,0,ksize=kernel_size)
        sobely = cv2.Sobel(diffuse,cv2.CV_64F,0,1,ksize=kernel_size)
        gradient = np.sqrt(np.square(sobelx)+np.square(sobely))
        gradient *= 1.0/gradient.max()
        return np.float32(gradient)
                
    def Nifti_Loader(self, dataroot, file):
        data_nii = nib.load(dataroot+file)
        data = np.array(data_nii.dataobj)
        data_crop = self.Center_Crop(data,im_size)
        return data_crop

    def ToTensor(self, image, mask):
        x_tensor = torch.tensor(image).type(torch.FloatTensor)
        y_tensor = transforms.functional.to_tensor(mask)
#        y_tensor = torch.squeeze(y_tensor,dim=0)
        return x_tensor, y_tensor    
        
    def __init__(self):
        self.pair = ()
        self.test_data = self.Nifti_Loader(dataroot,file_x)
        self.test_label = self.Nifti_Loader(dataroot,file_y)
        self.num = self.test_data.shape[2]
        
        for i in range(self.num):
            # Add gradient map as a prior
            x = self.test_data[:,:,i]
            edge_x = np.expand_dims(self.Grad_map(x,niter=15,kernel_size=3),axis=0)
            x = np.expand_dims(x,axis=0)
            x = np.concatenate((x,edge_x),axis=0)
            # find correspondance
            idx = np.int(np.floor(i/5))
            y = self.test_label[:,:,idx]
            self.pair = self.pair+((x,y),)
            
            if i/self.num == 0.5:
                print('50% finished.')
    
    def __len__(self):
        return self.num

    def __getitem__(self,idx):
        (image, mask) = self.pair[idx]
        x_tensor, y_tensor = self.ToTensor(image, mask)
        return x_tensor, y_tensor

test_loader = Data.DataLoader(dataset=Test_Dataset(), batch_size=batch_size, shuffle=True)

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
            nn.LeakyReLU(0.2)
            )
    
def up_block(in_channels,out_channels):
    return nn.Sequential(
            nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels,
                               kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
            )

def dual_block(n_ch):
     return nn.Sequential(
            nn.Conv2d(in_channels=n_ch, out_channels=n_ch,
                      kernel_size=3, stride=1, padding=1),
            nn.ELU(),
            nn.Conv2d(in_channels=n_ch, out_channels=n_ch,
                      kernel_size=3, stride=1, padding=1)
            )

class Generator(nn.Module):
    def __init__(self,gpu):
        super(Generator,self).__init__()
        self.gpu = gpu
        
        self.down_1 = down_block(2,64)       
        self.dual_d1 = dual_block(64)
        
        self.down_2 = down_block(64,128)
        self.dual_d2 = dual_block(128)
        
        self.down_3 = down_block(128,256)
        self.dual_d3 = dual_block(256)
        
        self.down_4 = down_block(256,512)
        self.dual_d4 = dual_block(512)
        
        self.up_1 = up_block(512,256)          # [256,64,64]
        self.dual_u1 = dual_block(256+256)     # [512,64,64]
        
        self.up_2 = up_block(256+256,256)      # [256,128,128]
        self.dual_u2 = dual_block(128+256)     # [384,128,128]
        
        self.up_3 = up_block(128+256,128)      # [128,256,256]
        self.dual_u3 = dual_block(64+128)      # [192,256,256]
        
        self.up_4 = up_block(64+128,64)        # [64,512,512]
        self.dual_u4 = dual_block(64)          # [64,512,512] 
        
        self.up_0 = up_block(64,64)
        self.Out = nn.Sequential(
                nn.Conv2d(in_channels=128,out_channels=64,kernel_size=1),
                nn.Conv2d(in_channels=64,out_channels=1,kernel_size=1),
                nn.Tanh()
                )
        
        self.relu = nn.ReLU()
        
    def forward(self,input):
        # Downwards
        x = self.down_1(input)
        layer_1 = torch.add(x,self.dual_d1(x))  # [64,256,256]
        x = self.down_2(layer_1)
        layer_2 = torch.add(x,self.dual_d2(x))  # [128,128,128]
        x = self.down_3(layer_2)
        layer_3 = torch.add(x,self.dual_d3(x))  # [256,64,64]
        x = self.down_4(layer_3)
        layer_4 = torch.add(x,self.dual_d4(x))  # [512,32,32]
    
        # Upwards
        x = self.up_1(layer_4)                 # [256,64,64] 
        x = torch.cat([x,layer_3],dim=1)       # [256+256,64,64]
        x = torch.add(x,self.dual_u1(x))        
        x = self.relu(x)
        
        x = self.up_2(x)                       # [256,128,128]
        x = torch.cat([x,layer_2],dim=1)       # [128+256,128,128]
        x = torch.add(x,self.dual_u2(x))       
        x = self.relu(x)
        
        x = self.up_3(x)                       # [128,256,256]
        x = torch.cat([x,layer_1],dim=1)       # [64+128,256,256]
        x = torch.add(x,self.dual_u3(x))
        x = self.relu(x)
        
        x = self.up_4(x)                       # [64,512,512]
        x = torch.add(x,self.dual_u4(x))
        x = self.relu(x)
        
        x = torch.cat([x,self.up_0(layer_1)],dim=1)
        output = self.Out(x)
        
        return output
        
netG = Generator(gpu).to(device)
netG.apply(weight_init)

#%%
modelroot = '/home/hud4/Desktop/2020/Model/'
G_name = 'Multiscale_cGAN_generator.pt'

netG.load_state_dict(torch.load(modelroot+G_name))

#%% Run the model
with torch.no_grad():
    for step, [x,y] in enumerate(test_loader):
        test_x = Variable(x).to(device)
        fake = netG(test_x).detach().cpu().numpy()

        gt = y.detach().cpu().numpy()
        img = x.detach().cpu().numpy()
            
        img_fake = np.transpose(fake[0,0,:,:])
        img_gt = np.transpose(gt[0,0,:,:])
        img_x = np.transpose(img[0,0,:,:])
            
        plt.figure(figsize=(15,8))
        plt.axis("off")
        plt.subplot(1,3,2),plt.imshow(img_fake)
        plt.subplot(1,3,3),plt.imshow(img_gt)
        plt.subplot(1,3,1),plt.imshow(img_x)
        plt.show()
        





