# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 11:01:10 2020

@author: hudew
"""

import sys
sys.path.insert(0,'C:\\Users\\hudew\\OneDrive\\桌面\\Denoise\\')
import MyFunctions

import numpy as np
import os
import cv2
import time
import pickle
import matplotlib.pyplot as plt
from medpy.filter.smoothing import anisotropic_diffusion

global radius
radius = 7
'''
Channel 1: original noisy B-scan
Channel 2: self-fused B-scan
Channel 3: gradient map

Due to radius contrain, the self-fused volume will have only 486 slices while 
the frame registration result will contain all 500 B-scans 
'''

root = 'E:\\Denoise_Train_Data\\'
v_name = 'Retina1_ONH_SNR_96_2'
file_list = []
for file in os.listdir(root):
    if file.startswith(v_name):
        file_list.append(file)

#%%
# Load the self-fused volume and get gradient map
Va = MyFunctions.nii_loader(root+file_list[1])
Vb = MyFunctions.nii_loader(root+file_list[2])
V_sf = np.concatenate([Va,Vb],axis=0)
del Va, Vb

# Load the paired x-y pickle file and crop to [r:-r]
with open(root+file_list[0],'rb') as f:
    pack_xy = pickle.load(f)
pack_xy = pack_xy[radius:-radius]
train_pair = ()

# Compute Gradient
def Sobel(img, kernel_size):
    sobelx = cv2.Sobel(img, cv2.CV_64F,1,0,ksize=kernel_size)
    sobely = cv2.Sobel(img, cv2.CV_64F,0,1,ksize=kernel_size)
    gradient = np.sqrt(np.square(sobelx),np.square(sobely))
    gradient *= 225.0/gradient.max()
    return np.float32(gradient)

t1 = time.time()
for i in range(len(pack_xy)):
    x = np.zeros([2,1024,512],dtype=np.float32)
    
    img_sf = V_sf[i,:,:] 
    diffuse = anisotropic_diffusion(img_sf,niter=30,option=2).astype(np.float32)
    gradient = Sobel(diffuse,3)
    
    x[0,:,:] = pack_xy[i][0]
    x[1,:,:] = gradient
    
    train_pair = train_pair+((x,pack_xy[i][1]),)
t2 = time.time()
print('time: {} min'.format((t2-t1)/60))

#%%
with open(root+v_name+'_pair.pickle','wb') as f:
    pickle.dump(train_pair,f)