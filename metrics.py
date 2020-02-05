#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 14:51:57 2020

@author: hud4
"""

import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

root = '/home/hud4/Desktop/'
noisy_file = 'noisy_125.nii.gz'
aver_file = 'average_125.nii.gz'
denoise_file = 'denoised_125_Bscan.nii.gz'

noisy_nii = nib.load(root+noisy_file)
noisy_img = np.array(noisy_nii.dataobj)/255.

aver_nii = nib.load(root+aver_file)
aver_img = np.array(aver_nii.dataobj)

denoise_nii = nib.load(root+denoise_file)
denoise_img = np.array(denoise_nii.dataobj)

#%%
slc = 250

im0 = noisy_img[100:300,200:400,slc]
im1 = aver_img[100:300,200:400,slc]
im2 = denoise_img[100:300,200:400,slc]

plt.figure(figsize=(15,12))
plt.subplot(1,3,1),plt.imshow(im0,cmap='gray')
plt.subplot(1,3,2),plt.imshow(im1,cmap='gray')
plt.subplot(1,3,3),plt.imshow(im2,cmap='gray')

#%%   SNR
snr_0 = np.mean(im0)/np.std(im0)
snr_1 = np.mean(im1)/np.std(im1)
snr_2 = np.mean(im2)/np.std(im2)

#%%   PSNR
import math
MSE = np.square(np.subtract(im1,im2)).mean()
PSNR = 10*math.log10(np.square(im2.max())/MSE)

#%%   SSIM
from skimage.measure import compare_ssim

(score,diff) = compare_ssim(im0,im1,full=True)

#%%   CNR
def CNR(Sa,Sb):
    diff = abs(Sa.mean()-Sb.mean())
    std = np.std(Sb)
    return diff/std

pnois_0 = noisy_img[:100,:100,slc]
pnois_1 = aver_img[:100,:100,slc]
pnois_2 = denoise_img[:100,:100,slc]

psig_0 = noisy_img[100:150,200:250,slc]
psig_1 = aver_img[100:150,200:250,slc]
psig_2 = denoise_img[100:150,200:250,slc] 


