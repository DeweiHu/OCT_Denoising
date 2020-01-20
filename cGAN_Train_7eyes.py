#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 23:06:33 2020

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
from torch.optim.lr_scheduler import StepLR
from torch.autograd import Variable

print('Loading data...')

dataroot = '/home/hud4/Desktop/2020/Data/'
train_x = 'train_x.npy'
train_y = 'train_y.npy'
test_x = 'test_x.npy'
test_y = 'test_y.npy'

#%% Train dataset and dataloader formation
print('Creating dataset...')

batch_size = 1
gpu = 1

class MyDataset(Data.Dataset):

    def cw90(self, img):
        [r,c] = img.shape
        opt = np.zeros([c,r])
        for i in range(r):
            vector = np.transpose(img[i,:])
            opt[:,r-i-1] = vector
        return np.uint8(opt)
    
    def ToTensor(self, image, mask):
        x_tensor = torch.tensor(image).type(torch.FloatTensor)
        y_tensor = transforms.functional.to_tensor(mask)
#        y_tensor = torch.squeeze(y_tensor,dim=0)
        return x_tensor, y_tensor    
        
    def __init__(self, x_dir, y_dir):
        self.pair = ()
        self.train_data = np.load(x_dir)      #[3500,3,512,512]
        self.train_label = np.load(y_dir)     #[512,512,3500]
        self.num = self.train_data.shape[0]
        
        for i in range(self.num):
            # Add gradient map as a prior
            x = self.train_data[i,:,:,:]
            y = self.cw90(self.train_label[:,:,i])
            self.pair = self.pair+((x,y),)

    def __len__(self):
        return self.num

    def __getitem__(self,idx):
        (image, mask) = self.pair[idx]
        x_tensor, y_tensor = self.ToTensor(image, mask)
        return x_tensor, y_tensor
    
train_loader = Data.DataLoader(dataset=MyDataset(dataroot+train_x,dataroot+train_y), 
                               batch_size=batch_size, shuffle=True)

test_loader = Data.DataLoader(dataset=MyDataset(dataroot+test_x,dataroot+test_y),
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
netG.apply(weight_init)

#%% Discriminator Architecture
class Discriminator(nn.Module):
    def __init__(self,gpu):
        super(Discriminator,self).__init__()
        self.gpu = gpu
        self.In = nn.Sequential(
                nn.Conv2d(in_channels=4,out_channels=64,kernel_size=4,stride=2,padding=1),
                nn.LeakyReLU(0.2),
                )
        self.main = nn.Sequential(
                nn.Conv2d(in_channels=64,out_channels=128,kernel_size=4,stride=2,padding=1),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(0.2),
                
                nn.Conv2d(in_channels=128,out_channels=256,kernel_size=4,stride=2,padding=1),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(0.2),
                
                nn.Conv2d(in_channels=256,out_channels=512,kernel_size=4,stride=2,padding=1),
                nn.BatchNorm2d(512),
                nn.LeakyReLU(0.2),
                )
        self.Out = nn.Sequential(
                nn.Linear(batch_size*32*32*512,100),
                nn.Linear(100,batch_size),
                nn.Sigmoid()
                )
    def forward(self,input):
        x = self.In(input)
        x = self.main(x)
        x = x.view(-1)        # Vectorize to a column vector
        output = self.Out(x)
        return output

netD = Discriminator(gpu).to(device)
netD.apply(weight_init)

#%% Loss functions and optimizers
BCE_loss = nn.BCELoss()
L1_loss = nn.L1Loss()
MSE_loss = nn.MSELoss()

real_label = 1
fake_label = 0

# Adam optimizers
beta1 = 0.5
lr = 0.00002
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1,0.999))
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1,0.999))
schedulerG = StepLR(optimizerG, step_size=1, gamma=0.5)
schedulerD = StepLR(optimizerD, step_size=1, gamma=0.5)

#%% Training 
import time

img_list = []
G_losses = []
D_losses = []
num_epoch = 10

alpha = 5

print('Training process start:')

t1 = time.time()
for epoch in range(num_epoch):
    for step,[train_x,train_y] in enumerate(train_loader):
        
        #####  {Part 1} Discriminator: max{log(D(x))+log(1-D(G(z)))}  #####
        # The discriminator is exclusively pre-trained 
        netD.zero_grad()
        
        x = Variable(train_x).to(device)
        y = Variable(train_y).to(device)
        label = torch.full((batch_size,), real_label, device=device)
        
        # [1] Pre-train the discriminator with real image
        real_pair = torch.cat([x,y],dim=1)
        output = netD(real_pair).view(-1)
        
        D_error_real = BCE_loss(output,label)
        D_error_real.backward()
        D_x = output.mean().item()    # D(x)
        
        # [2] Pre-train the discriminator with fake image
        fake_y = netG(x)
        label.fill_(fake_label)
        
        fake_pair = torch.cat([x,fake_y],dim=1)
        output = netD(fake_pair.detach()).view(-1)  # detach from netG, only update netD
        
        D_error_fake = BCE_loss(output,label)
        D_error_fake.backward()
        D_G_z1 = output.mean().item()  # D(G(z))
        
        # Sum up the loss and gradient
        D_error = D_error_real+D_error_fake
        optimizerD.step()

        #####  {Part 2} Generator: max{log(D(G(z)))}  #####
        netG.zero_grad()
        label.fill_(real_label)    # Generator want D(G(z)) close to 1
        
        output = netD(fake_pair).view(-1)
        
        G_error = BCE_loss(output,label)+alpha*L1_loss(fake_y,y)+MSE_loss(fake_y,y)
        G_error.backward()
        D_G_z2 = output.mean().item()
        optimizerG.step()
        
        # Viusualization
        if step % 300 == 0:
 #           print(D_error_fake)
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  %(epoch, num_epoch, step, len(train_loader), D_error.item(), 
                    G_error.item(), D_x, D_G_z1, D_G_z2))
            
        if step == 0:
            # Train data denoising
            denoise_train = netG(x).detach().cpu().numpy()
            noise_train = train_x.numpy()
            avg_train = train_y.numpy()
            
            # Test data denoising
            for test_x, test_y in test_loader:
                ipt_x = test_x.to(device)
                denoise_test = netG(ipt_x).detach().cpu().numpy()
                noise_test = test_x.numpy()
                avg_test = test_y.numpy()
                
            plt.figure(figsize=(18,15))
            plt.subplot(2,3,1),plt.imshow(noise_train[0,0,:,:]),plt.title('noisy')
            plt.subplot(2,3,2),plt.imshow(denoise_train[0,0,:,:]),plt.title('denoised')
            plt.subplot(2,3,3),plt.imshow(avg_train[0,0,:,:]),plt.title('5-average')
            
            plt.subplot(2,3,4),plt.imshow(noise_test[0,0,:,:])
            plt.subplot(2,3,5),plt.imshow(denoise_test[0,0,:,:])
            plt.subplot(2,3,6),plt.imshow(avg_test[0,0,:,:])
            plt.show()
            
            
        
        G_losses.append(G_error.item())
        D_losses.append(D_error.item())

t2 = time.time()
print('Time used:',(t2-t1)/60,' min')

#%% Save model as GPU version
modelroot = '/home/hud4/Desktop/2020/Model/'
G_name = 'EP_cGAN_generator.pt'
D_name = 'Ep_cGAN_discriminator.pt'
torch.save(netG.state_dict(), modelroot+G_name)
torch.save(netD.state_dict(), modelroot+D_name)
    
    