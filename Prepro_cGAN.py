#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 15:48:36 2020

@author: hud4
"""

import nibabel as nib
import numpy as np
import os
import cv2
import time
from medpy.filter.smoothing import anisotropic_diffusion
import matplotlib.pyplot as plt


global num_frame, num_volume
num_frame = 5
num_volume = 7

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

    def cw90(self, img):
        [r,c] = img.shape
        opt = np.zeros([c,r])
        for i in range(r):
            vector = np.transpose(img[i,:])
            opt[:,r-i-1] = vector
        return opt

    def Sobel(self, img, kernel_size):
        sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=kernel_size)
        sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=kernel_size)
        gradient = np.sqrt(np.square(sobelx)+np.square(sobely))
        gradient *= 255.0/gradient.max()
        return np.float32(gradient)

    def __init__(self, dataroot, file, im_size):
        # Load in nii as a 3d matrix
        data_nii = nib.load(dataroot+file)
        data = np.array(data_nii.dataobj)
        num = np.int(data.shape[2]/num_frame)
        self.pack = np.zeros([num,3,im_size,im_size]).astype(np.float32)
        
        t1 = time.time()
        
        for i in range(num):
            data_crop = self.Center_Crop(self.cw90(data[:,:,i*num_frame]),im_size)
            # get gradient map
            diffuse_1 = anisotropic_diffusion(data_crop,niter=15,option=2).astype(np.uint8)
            diffuse_2 = anisotropic_diffusion(data_crop,niter=30,option=2).astype(np.uint8)
            
            gradient_1 = self.Sobel(diffuse_1,3)
            gradient_2 = self.Sobel(diffuse_2,3)
            
            self.pack[i,0,:,:] = data_crop
            self.pack[i,1,:,:] = gradient_1
            self.pack[i,2,:,:] = gradient_2
            
        t2 = time.time()
        print('Processing time: ',np.int(t2-t1),'s')
        
        
dataroot = '/sdc/MiceData/'
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

#%%
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

x_all = np.zeros([500*num_volume,3,512,512]).astype(np.float32)
y_all = np.zeros([512,512,500*num_volume]).astype(np.uint8)

for i in range(num_volume):    
    volume = Packer(dataroot,x_volume[i],512)
    x_all[500*i:500*(i+1),:,:,:] = volume.pack
    
    y_nii = nib.load(dataroot+y_volume[i])
    y_all[:,:,500*i:500*(i+1)] = Crop3d(np.array(y_nii.dataobj),512)
    print(i)

#%% save
saveroot = '/home/hud4/Desktop/2020/Data/'
x_name = 'volume_x'
y_name = 'volume_y'

np.save(saveroot+x_name,x_all)
np.save(saveroot+y_name,y_all)


