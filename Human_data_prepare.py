#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 12:03:55 2020

@author: hud4
"""

from skimage import io
import matplotlib.pyplot as plt
import numpy as np
import os

def Rescale(img):
    val_max = img.max()
    val_min = img.min()
    val_range = val_max-val_min
    opt = (img-val_min)/val_range
    return opt

saveroot = '/home/hud4/Desktop/2020/Human/'
filelist = []

for file in os.listdir(saveroot): 
    if file.endswith(".tif"):
        filelist.append(file)

filelist.sort()
#%% Train data        
num = 12
train_list = filelist[:num]
train_x = np.zeros([500*num,1024,512],dtype=np.float32)
train_y = np.zeros([500*num,1024,512],dtype=np.float32)

for idx in range(num):
    print('file {} processing...'.format(idx+1))
    file = train_list[idx]
    im = Rescale(io.imread(saveroot+file))  #[2500,1024,500]
    for i in range(len(im)):
        if i % 5 == 0:
            aver = 1/5*np.sum(im[i:i+5,:,:],axis=0)
            train_x[500*idx+int(i/5),:,:500] = im[i,:,:]*255
            train_y[500*idx+int(i/5),:,:500] = aver
            
np.save(saveroot+'Human_train_x',train_x)
np.save(saveroot+'Human_train_y',train_y)    

#%% Test data
for idx in range(len(filelist)):

    test_file = filelist[idx]
    test_x = np.zeros([500,1024,512],dtype=np.float32)
    test_y = np.zeros([500,1024,512],dtype=np.float32)
    
    im = Rescale(io.imread(saveroot+test_file))  #[2500,1024,500]
    for i in range(len(im)):
        if i % 5 == 0:
            aver = 1/5*np.sum(im[i:i+5,:,:],axis=0)
            test_x[int(i/5),:,:500] = im[i,:,:]*255
            test_y[int(i/5),:,:500] = aver
            
    np.save(saveroot+'{}_x'.format(test_file),test_x)
    np.save(saveroot+'{}_y'.format(test_file),test_y)    


#%%
plt.figure(figsize=(12,12))
plt.subplot(1,2,1),plt.imshow(train_x[10,:,:],cmap='gray')        
plt.subplot(1,2,2),plt.imshow(train_y[10,:,:],cmap='gray')
