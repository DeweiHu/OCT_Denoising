# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 10:25:48 2020

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
        
    def __init__(self, dataroot):
        with open(dataroot,'rb') as f:
            self.pair = pickle.load(f)
        
    def __len__(self):
        return len(self.pair)

    def __getitem__(self,idx):
        (image, mask) = self.pair[idx]
        x_tensor, y_tensor = self.ToTensor(image,
                                MyFunctions.ImageRescale(mask,[0,1]))
        return x_tensor, y_tensor
    
test_loader = Data.DataLoader(dataset=MyDataset(root+'Retina2_test.pickle'), 
                               batch_size=batch_size, shuffle=True)

device = torch.device("cuda:0" if( torch.cuda.is_available() and gpu>0 ) else "cpu")

#%%
#for step,[x,y] in enumerate(test_loader):
#    pass

#%% Loader generator arch
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
        self.up_01 = up_block(128,64)
        
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
        
        att = torch.cat([layer_1,self.up_01(layer_2)],dim=1)
        x = torch.cat([x,self.up_01(att)],dim=1)
        output = self.Out(x)
        
        return output
        
model = Generator(gpu).to(device)

modelroot = 'E:\\Model\\'
name = 'SF_cGAN_generator.pt'
model.load_state_dict(torch.load(modelroot+name))

#%% Test
for step,[img,mask] in enumerate(test_loader):
    
    model.eval()
    with torch.no_grad():
        x = Variable(img).to(device)
        pred_y = model(x)
        
        pred_y = pred_y.detach().cpu().numpy()
        x = img.numpy()
        y = mask.numpy()
    
        if step % 40 == 0:
            plt.figure(figsize=(18,15))
            plt.subplot(1,4,1),plt.axis('off'),
            plt.imshow(x[0,0,:,:],cmap='gray'),plt.title('noisy input')
            plt.subplot(1,4,2),plt.axis('off'),
            plt.imshow(x[0,1,:,:],cmap='gray'),plt.title('gradmap')
            plt.subplot(1,4,3),plt.axis('off'),
            plt.imshow(y[0,0,:,:],cmap='gray'),plt.title('5-average')
            plt.subplot(1,4,4),plt.axis('off'),
            plt.imshow(pred_y[0,0,:,:],cmap='gray'),plt.title('denoise')
            plt.show()



