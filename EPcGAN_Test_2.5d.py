#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 20:28:24 2020

@author: hud4
"""

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision.transforms as transforms
from torch.autograd import Variable

print('Loading data...')

global axis
axis = 0

dataroot = '/home/hud4/Desktop/2020/Data/' 
file_x = 'v125_x_ax{}.npy'.format(axis)
file_y = 'v125_y_ax{}.npy'.format(axis)
#%% Train dataset and dataloader formation
print('Creating dataset...')

batch_size = 1
gpu = 1

class MyDataset(Data.Dataset):

    def ToTensor(self, image, mask):
        x_tensor = torch.tensor(image).type(torch.FloatTensor)
        y_tensor = transforms.functional.to_tensor(mask)
#        y_tensor = torch.squeeze(y_tensor,dim=0)
        return x_tensor, y_tensor    
        
    def __init__(self, x_dir, y_dir, axis):
        self.pair = ()
        self.train_data = np.load(x_dir)      #[4096,3,512,500]
        self.train_label = np.load(y_dir)     #[4096,512,500] or [512,4096,500]
        self.num = self.train_data.shape[0]
        
        for i in range(self.num):
            # pre-define the shape of input
            x = np.zeros([3,512,512],dtype=np.float32)
            y = np.zeros([512,512],dtype=np.uint8)
            # Add gradient map as a prior
            x[:,:,:500] = self.train_data[i,:,:,:]
            
            if axis == 0:
                y[:,:500] = self.train_label[i,:,:]
            elif axis == 1:
                y[:,:500] = self.train_label[:,i,:]
            else:
                print('Axis unrecognized.')
                    
            self.pair = self.pair+((x,y),)

    def __len__(self):
        return self.num

    def __getitem__(self,idx):
        (image, mask) = self.pair[idx]
        x_tensor, y_tensor = self.ToTensor(image, mask)
        return x_tensor, y_tensor
    
test_loader = Data.DataLoader(dataset=MyDataset(dataroot+file_x,dataroot+file_y,axis),
                              batch_size=batch_size, shuffle=False)

device = torch.device("cuda:0" if( torch.cuda.is_available() and gpu>0 ) else "cpu")


#%% Generator Architecture
print('Initializing model...')

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
        
        self.down_1 = down_block(3,64)       
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

#%%
modelroot = '/home/hud4/Desktop/2020/Model/'

if axis == 0:
    G_name = 'EP_cGAN_generator_ax0.pt'
elif axis == 1:
    G_name = 'EP_cGAN_generator_ax1.pt'
else:
    print('Axis wrong.')
    
netG.load_state_dict(torch.load(modelroot+G_name))

shape = [512,512,500]
denoise_volume = np.zeros(shape).astype(np.float32)
average_volume = np.zeros(shape).astype(np.float32)
noisy_volume = np.zeros(shape).astype(np.float32)

#%% Run the model
import time

def cw90(img):
    [r,c] = img.shape
    opt = np.zeros([c,r])
    for i in range(r):
        vector = np.transpose(img[i,:])
        opt[:,r-i-1] = vector
    return opt

t1 = time.time()
with torch.no_grad():
    for step, [x,y] in enumerate(test_loader):
        test_x = Variable(x).to(device)
        
        denoise = netG(test_x).detach().cpu().numpy()
        average = y.numpy()
        noisy = x.detach().cpu().numpy()
        
        # change sequence for different axis 
        if axis == 0:
            denoise_volume[step,:,:] = np.flipud(denoise[0,0,:,:500])
        elif axis == 1:
            denoise_volume[:,step,:] = np.flipud(denoise[0,0,:,:500])
        else:
            print('Axis wrong.')
        
        if step % 30 == 0:
            plt.figure(figsize=(15,8))
            plt.subplot(1,3,1),plt.imshow(cw90(noisy[0,0,:,:]),cmap='gray')
            plt.subplot(1,3,2),plt.imshow(cw90(denoise[0,0,:,:]),cmap='gray')
            plt.subplot(1,3,3),plt.imshow(cw90(average[0,0,:,:]),cmap='gray')
            plt.show()
t2 = time.time()

print('Time used: ', t2-t1,'s')
#%% Save the volume as nii
import os
def save_nii(volume,path,filename):
    output = nib.Nifti1Image(volume,np.eye(4))
    nib.save(output,os.path.join(path,filename))
    
save_nii(denoise_volume,'/home/hud4/Desktop/2020/','denoised_125_ax{}.nii.gz'.format(axis))