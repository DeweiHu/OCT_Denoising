#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 20:02:42 2020

@author: hud4
"""

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from medpy.filter.smoothing import anisotropic_diffusion
from skimage import feature

global dataroot, file_x, file_y
dataroot = '/home/hud4/Desktop/2020/Data/'
file_x = '2019-09-18-001_OD_V_6x6_0_0000043_structure.nii'
file_y = '2019-09-18-001_OD_V_6x6_0_0000043_structure_5avg.nii'

def cw90(img):
    [r,c] = img.shape
    opt = np.zeros([c,r])
    for i in range(r):
        vector = np.transpose(img[i,:])
        opt[:,r-i-1] = vector
    return opt


x_nii = nib.load(dataroot+file_x)
x = np.array(x_nii.dataobj)

y_nii = nib.load(dataroot+file_y)
y = np.array(y_nii.dataobj)
#        
img_x = cw90(x[:,:,1500])
img_y = cw90(y[:,:,289])
img_smooth_1 = anisotropic_diffusion(img_y,niter=30,option=1)
img_smooth_2 = anisotropic_diffusion(img_x,niter=12,option=2)
img_smooth_3 = anisotropic_diffusion(img_y,niter=30,option=3)

plt.figure(figsize=(18,12))
plt.subplot(1,3,1),plt.imshow(img_x)
plt.subplot(1,3,2),plt.imshow(img_y)
plt.subplot(1,3,3),plt.imshow(img_smooth_2)

#%% Column-out

n_cl = 300
x = img_x[:,n_cl]
y = img_y[:,n_cl]
y_smooth = img_smooth_2[:,n_cl]

plt.figure(figsize=(15,8))
plt.plot(np.arange(0,600),y_smooth,'r',linewidth=4.0)
plt.plot(np.arange(0,600),y,'b',linewidth=1.0)


#%% Canny edge detection
import cv2
laplacian_smooth = cv2.Laplacian(np.uint8(img_smooth_2),cv2.CV_64F)

im = np.uint8(img_smooth_2)

sobelx = cv2.Sobel(im,cv2.CV_64F,1,0,ksize=3)
sobely = cv2.Sobel(im,cv2.CV_64F,0,1,ksize=3)
gradient_mag = np.sqrt(np.square(sobelx)+np.square(sobely))
gradient_mag *= 1.0/gradient_mag.max()

plt.figure(figsize=(8,8))
plt.imshow(gradient_mag)

#plt.imshow(img_x)
