#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 20:59:41 2020

@author: hud4
"""

import nibabel as nib
import numpy as np
import os
import cv2
import time
from medpy.filter.smoothing import anisotropic_diffusion
import matplotlib.pyplot as plt

dataroot = '/sdc/MiceData/'
saveroot = '/home/hud4/Desktop/2020/Data/' 

global num_frame, num_volume
num_frame = 5
num_volume = 8
        
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

class DataPacker:
    
    def Slicer(self, volume, num_frame):
        target_slice = np.int(volume.shape[2]/num_frame)
        opt = np.zeros([volume.shape[0],volume.shape[1],target_slice],dtype=np.uint8)
        for i in range(volume.shape[2]):
            if i % num_frame == 0:
                opt[:,:,int(np.floor(i/num_frame))] = volume[:,:,i]
        return opt
    
    def Sobel(self, img, kernel_size):
        sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=kernel_size)
        sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=kernel_size)
        gradient = np.sqrt(np.square(sobelx)+np.square(sobely))
        gradient *= 255.0/gradient.max()
        return np.float32(gradient)
    
    def __init__(self, dataroot, file, axis):
        # Load in nii as a 3d matrix
        data_nii = nib.load(dataroot+file)
        data = np.array(data_nii.dataobj)
        # Slicer
        single_frame_volume = self.Slicer(data,5)
        # 3d crop
        crop = Crop3d(single_frame_volume,512)
        # Define the output
        self.pack = np.zeros([512,3,512,500],dtype=np.float32)
        # Which dimension
        if axis == 0:
            for i in range(crop.shape[axis]):
                img = crop[i,:,:]
                # get gradient map
                diffuse_1 = anisotropic_diffusion(img,niter=15,option=2).astype(np.uint8)
                diffuse_2 = anisotropic_diffusion(img,niter=30,option=2).astype(np.uint8)
            
                gradient_1 = self.Sobel(diffuse_1,3)
                gradient_2 = self.Sobel(diffuse_2,3)
                
                self.pack[i,0,:,:] = img
                self.pack[i,1,:,:] = gradient_1
                self.pack[i,2,:,:] = gradient_2
            print('Axis 0 finished.')
            
        elif axis == 1:
            for i in range(crop.shape[axis]):
                img = crop[:,i,:]
                # get gradient map
                diffuse_1 = anisotropic_diffusion(img,niter=15,option=2).astype(np.uint8)
                diffuse_2 = anisotropic_diffusion(img,niter=30,option=2).astype(np.uint8)
            
                gradient_1 = self.Sobel(diffuse_1,3)
                gradient_2 = self.Sobel(diffuse_2,3)
                
                self.pack[i,0,:,:] = img
                self.pack[i,1,:,:] = gradient_1
                self.pack[i,2,:,:] = gradient_2
            print('Axis 1 finished.')
            
        else: 
            print('Error: Axis unrecognized.')

#%% Data package 
for axis in range(2):
    x = np.zeros([512*num_volume,3,512,500],dtype=np.float32)
    y = np.zeros([512*num_volume,512,500]).astype(np.uint8)

    for i in range(num_volume):    
        volume = DataPacker(dataroot,x_volume[i],axis)
        x[512*i:512*(i+1),:,:,:] = volume.pack
        
        y_nii = nib.load(dataroot+y_volume[i])
        y[512*i:512*(i+1),:,:] = Crop3d(np.array(y_nii.dataobj),512)
        print(i)

    # save training data
    x_name = 'train_x_ax{}'.format(axis)
    y_name = 'train_y_ax{}'.format(axis)

    np.save(saveroot+x_name,x)
    np.save(saveroot+y_name,y)   

    print('Training data for axis {} has done.'.format(axis))

    # Testing data package 
    test_volume_x = x_list[num_volume-1:-1]
    test_volume_y = y_list[num_volume-1:-1]

    test_x = np.zeros([512*(len(x_list)-num_volume),3,512,500],dtype=np.float32)
    test_y = np.zeros([512*(len(y_list)-num_volume),512,500],dtype=np.uint8)

    for i in range(len(x_list)-num_volume):    
        volume = DataPacker(dataroot,test_volume_x[i],axis)
        test_x[512*i:512*(i+1),:,:,:] = volume.pack
    
        y_nii = nib.load(dataroot+test_volume_y[i])
        test_y[512*i:512*(i+1),:,:] = Crop3d(np.array(y_nii.dataobj),512)
        print(i)
    
    x_name = 'test_x_ax{}'.format(axis)
    y_name = 'test_y_ax{}'.format(axis)
    
    np.save(saveroot+x_name,test_x)
    np.save(saveroot+y_name,test_y)

    print('Testing data for axis {} has done.'.format(axis))



