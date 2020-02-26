#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 13:44:27 2020

@author: hud4
"""

import nibabel as nib
import numpy as np
import os
import cv2
import time
from medpy.filter.smoothing import anisotropic_diffusion
import matplotlib.pyplot as plt
import time

global num_frame, num_volume
num_frame = 5
num_volume = 8

dataroot = '/sdc/MiceData/'
saveroot = '/home/hud4/Desktop/2020/Data/'

test_x_root = '/sdc/MoreData/2019-07-03-001_OD_V_6x6_0_0000053_structure.nii' 
test_y_root = '/sdc/MoreData/2019-07-03-001_OD_V_6x6_0_0000053_structure_5avg.nii' 

def Crop3d(data, im_size):
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

class Packer:
    
    def Center_Crop(self, img, im_size):
        [r,c] = img.shape
        data_crop = np.zeros([im_size,im_size])
        left = np.int((c-im_size)/2)
        right = c-left
        top = np.int((r-im_size)/2)
        bottom = r-top
        data_crop = img[top:bottom,left:right]
        return data_crop

    def Sobel(self, img, kernel_size):
        sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=kernel_size)
        sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=kernel_size)
        gradient = np.sqrt(np.square(sobelx)+np.square(sobely))
        gradient *= 255.0/gradient.max()
        return np.float32(gradient)

    def __init__(self, dataroot, file_x, file_y, im_size):
        
        # Load in nii as a 3d matrix
        data_x_nii = nib.load(dataroot+file_x)
        data_x = np.array(data_x_nii.dataobj)
        
        data_y_nii =nib.load(dataroot+file_y)
        data_y = np.array(data_y_nii.dataobj)
        num = data_y.shape[2]
        
        self.pack = np.zeros([num,3,im_size,im_size]).astype(np.float32)
        
        t1 = time.time()
        
        for i in range(num):
            data_x_crop = self.Center_Crop(data_x[:,:,i*num_frame],im_size)
            data_y_crop = self.Center_Crop(data_y[:,:,i],im_size)
            # get gradient map
            diffuse_1 = anisotropic_diffusion(data_x_crop,niter=15,option=2).astype(np.uint8)
            diffuse_2 = anisotropic_diffusion(data_x_crop,niter=30,option=2).astype(np.uint8)
            
            gradient_1 = self.Sobel(diffuse_1,3)
            gradient_2 = self.Sobel(diffuse_2,3)
            
            # Threshold
            bg_1 = gradient_1[400:500,0:100]
            bg_2 = gradient_2[400:500,0:100]
            gradient_1[gradient_1<bg_1.mean()+25] = 0
            gradient_2[gradient_2<bg_2.mean()+25] = 0
            
            self.pack[i,0,:,:] = data_x_crop
            self.pack[i,1,:,:] = gradient_1
            self.pack[i,2,:,:] = gradient_2
            
        t2 = time.time()
        print('Processing time: ',np.int(t2-t1),'s')

x_list = []
y_list = []

for file in os.listdir(dataroot):
    if file.endswith("structure.nii"):
        x_list.append(file)
    else:
        y_list.append(file)

x_list.sort()
y_list.sort()

x_volume = x_list[:num_volume]
y_volume = y_list[:num_volume]

#%%  Prepare training data
x_all = np.zeros([500*num_volume,3,512,512]).astype(np.float32)
y_all = np.zeros([512,512,500*num_volume]).astype(np.uint8)

for i in range(num_volume):    
    volume = Packer(dataroot,x_volume[i],y_volume[i],512)
    x_all[500*i:500*(i+1),:,:,:] = volume.pack
    
    y_nii = nib.load(dataroot+y_volume[i])
    y_all[:,:,500*i:500*(i+1)] = Crop3d(np.array(y_nii.dataobj),512)
    print(i)

# save training data
x_name = 'cross_train_x.npy'
y_name = 'cross_train_y.npy'

np.save(saveroot+x_name,x_all)
np.save(saveroot+y_name,y_all)

#%%  Prepare testing data
test_volume_x = x_list[num_volume-1:-1]
test_volume_y = y_list[num_volume-1:-1]

test_x = np.zeros([500*(len(x_list)-num_volume),3,512,512]).astype(np.float32)
test_y = np.zeros([512,512,500*(len(y_list)-num_volume)]).astype(np.uint8)

for i in range(len(x_list)-num_volume):
    volume = Packer(dataroot,test_volume_x[i],test_volume_y[i],512)
    test_x[500*i:500*(i+1),:,:,:] = volume.pack
    
    y_nii = nib.load(dataroot+test_volume_y[i])
    test_y[:,:,500*i:500*(i+1)] = Crop3d(np.array(y_nii.dataobj),512)
    print(i)

np.save(saveroot+'cross_test_x',test_x)
np.save(saveroot+'cross_test_y',test_y)

#%% Single nii testing
x = np.zeros([500,3,512,512],dtype=np.float32)
y = np.zeros([512,512,500],dtype=np.uint8)

x = Packer('/sdc/MoreData/','2019-07-03-001_OD_V_6x6_0_0000053_structure.nii',
           '2019-07-03-001_OD_V_6x6_0_0000053_structure_5avg.nii',512)
y_nii = nib.load('/sdc/MoreData/'+'2019-07-03-001_OD_V_6x6_0_0000053_structure_5avg.nii') 
y = Crop3d(np.array(y_nii.dataobj),512)

np.save(saveroot+'cross_mice_test_x.npy',x.pack)
np.save(saveroot+'cross_mice_test_y.npy',y)
