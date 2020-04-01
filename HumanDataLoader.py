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

def Sobel(img, kernel_size):
    sobelx = cv2.Sobel(img, cv2.CV_64F,1,0,ksize=kernel_size)
    sobely = cv2.Sobel(img, cv2.CV_64F,0,1,ksize=kernel_size)
    gradient = np.sqrt(np.square(sobelx)+np.square(sobely))
    gradient *= 255.0/gradient.max()
    return np.float32(gradient)

'''
Channel 1: original noisy B-scan
Channel 2: self-fused B-scan
Channel 3: gradient map

Due to radius contrain, the self-fused volume will have only 486 slices while 
the frame registration result will contain all 500 B-scans 
'''

root = 'E:\\Denoise_Train_Data\\'
volume_list = []
for vol in os.listdir(root):
    if vol.startswith('Retina1'):
        volume_list.append(vol)

train_pair = ()
for vol in range(len(volume_list)):
    '''
    x-y pair is saved in .pickle file (FR)
    self-fused volume is saved in _a.nii.gz and _b.nii.gz (SF)
    '''
    FR_dir = []
    SF_dir = []
    for file in os.listdir(root+volume_list[vol]):
        if file.endswith('.pickle'):
            FR_dir.append(root+volume_list[vol]+'\\'+file)
        else:
            SF_dir.append(root+volume_list[vol]+'\\'+file)
            
    # Load the self-fused volume and get gradient map
    Va = MyFunctions.nii_loader(SF_dir[0])
    Vb = MyFunctions.nii_loader(SF_dir[1])
    V_sf = np.concatenate([Va,Vb],axis=0)
    del Va, Vb    

    # Load the paired x-y pickle file and crop to [r:-r]
    with open(FR_dir[0],'rb') as f:
        pack_xy = pickle.load(f)
    pack_xy = pack_xy[radius:-radius]

    # Diffuse and compute gradient
    t1 = time.time()
    
    for i in range(len(pack_xy)):
        x = np.zeros([3,1024,512],dtype=np.float32)
        gradient = np.zeros([1024,512],dtype=np.float32)
        
        img_sf = V_sf[i,:,:] 
        diffuse = anisotropic_diffusion(img_sf,niter=20,option=2).astype(np.float32)
        gradient[:,:500] = Sobel(diffuse[:,:500],3)
        
        x[0,:,:] = pack_xy[i][0]
        x[1,:,:] = gradient
        x[2,:,:] = img_sf
        
        train_pair = train_pair+((x,pack_xy[i][1]),)
    
    t2 = time.time()
    print('volume %d. time consumption: %.4f min' 
          %(vol, (t2-t1)/60))

#%% 
#img = train_pair[400][0]
#plt.figure(figsize=(12,12))
#plt.subplot(1,3,1),plt.axis('off'),plt.imshow(img[0,:,:],cmap='gray')
#plt.subplot(1,3,2),plt.axis('off'),plt.imshow(img[1,:,:],cmap='gray')
#plt.subplot(1,3,3),plt.axis('off'),plt.imshow(img[2,:,:],cmap='gray')

#%%
with open(root+'Retina2_train.pickle','wb') as f:
    pickle.dump(train_pair,f)