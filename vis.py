#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 11:50:45 2020

@author: hud4
"""

from skimage import io
import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib

root = '/home/hud4/Desktop/regi_result/'

def nifti_load(dir):
    data_nii = nib.load(dir)
    data = np.array(data_nii.dataobj)
    return data

def cw90(img):
    [r,c] = img.shape
    opt = np.zeros([c,r])
    for i in range(r):
        vector = np.transpose(img[i,:])
        opt[:,r-i-1] = vector
    return opt

def ccw90(img):
    [r,c] = img.shape
    opt = np.zeros([c,r])
    for i in range(r):
        vector = np.transpose(img[:,i])
        opt[c-i-1,:] = vector
    return opt

im_y = nifti_load(root+'reg_y.nii.gz')
im_sm_1 = io.imread(root+'synthResult.tif')
#im_sm_2 = io.imread(root+'synthResult_ex.tif')

#%%
plt.figure(figsize=(18,15))
#plt.subplot(1,3,1),plt.imshow(np.fliplr(ccw90(im_x[13,:,:])),cmap='gray')
plt.subplot(1,3,1),plt.imshow(np.fliplr(ccw90(im_sm_1)),cmap='gray')
plt.subplot(1,3,2),plt.imshow(np.fliplr(ccw90(im_sm_2)),cmap='gray')
plt.subplot(1,3,3),plt.imshow(np.fliplr(ccw90(im_y[0,:,:])),cmap='gray')

#%%
plt.figure(figsize=(18,15))
#plt.subplot(1,3,1),plt.imshow(np.fliplr(ccw90(im_x[13,:,:])),cmap='gray')
plt.subplot(1,2,1),plt.imshow(np.fliplr(im_sm_1),cmap='gray')
plt.subplot(1,2,2),plt.imshow(np.fliplr(im_y[0,:,:]),cmap='gray')