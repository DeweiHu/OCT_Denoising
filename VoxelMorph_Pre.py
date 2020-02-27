#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 23:05:48 2020

@author: hud4
"""

import nibabel as nib
import numpy as np
import os
import cv2
import time
from medpy.filter.smoothing import anisotropic_diffusion
import matplotlib.pyplot as plt


global num_frame, im_size,num_volume
num_frame = 5
im_size = 512
num_volume = 1

dataroot = '/sdc/MiceData/'
saveroot = '/home/hud4/Desktop/2020/Data/'

x_list = []

for file in os.listdir(dataroot):
    if file.endswith("structure.nii"):
        x_list.append(file)

x_list.sort()

x_volume = x_list[:num_volume]


#%%
def nii_loader(dir,file):
    data_nii = nib.load(dir+file)
    data = np.array(data_nii.dataobj)
    return data

def Center_Crop(img, im_size):
        [r,c] = img.shape
        data_crop = np.zeros([im_size,im_size])
        left = np.int((c-im_size)/2)
        right = c-left
        top = np.int((r-im_size)/2)
        bottom = r-top
        data_crop = img[top:bottom,left:right]
        return data_crop

def PickFrame(volume,num_frame):
    size = volume.shape
    v = np.zeros([im_size,im_size,int(size[-1]/num_frame)],dtype=np.uint8)
    
    for i in range(size[-1]):
        if i % num_frame == 0:
            idx = int(i/num_frame)
            img = Center_Crop(volume[:,:,i],im_size)
            v[:,:,idx] = img
    return v

def cw90(img):
    [r,c] = img.shape
    opt = np.zeros([c,r])
    for i in range(r):
        vector = np.transpose(img[i,:])
        opt[:,r-i-1] = vector
    return opt
            
data = nii_loader(dataroot,x_list[0])
v = PickFrame(data, num_frame)

plt.imshow(cw90(v[:,:,0]),cmap='gray')
