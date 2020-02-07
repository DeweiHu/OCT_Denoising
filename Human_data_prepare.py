# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 13:56:20 2020

@author: hudew
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

root = "C:\\Users\\hudew\\OneDrive\\桌面\\Denoise\\"
filelist = []

for file in os.listdir(root): 
    if file.endswith(".tif"):
        filelist.append(file)
        
num = 12
train_list = filelist[:num]
train_x = np.zeros([500*num,1024,512],dtype=np.float32)
train_y = np.zeros([500*num,1024,512],dtype=np.float32)

for idx in range(num):
    print('file {} processing...'.format(idx+1))
    file = train_list[idx]
    im = Rescale(io.imread(root+file))  #[2500,1024,500]
    for i in range(len(im)):
        if i % 5 == 0:
            aver = 1/5*np.sum(im[i:i+5,:,:],axis=0)
            train_x[500*idx+int(i/5),:,:500] = im[i,:,:]*255
            train_y[500*idx+int(i/5),:,:500] = aver
            
np.save(root+'train_x',train_x)
np.save(root+'train_y',train_y)    
#%%
plt.figure(figsize=(12,12))
plt.subplot(1,2,1),plt.imshow(train_x[10,:,:],cmap='gray')        
plt.subplot(1,2,2),plt.imshow(train_y[10,:,:],cmap='gray')
