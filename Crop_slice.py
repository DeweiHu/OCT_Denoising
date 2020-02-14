#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 16:43:44 2020

@author: dewei
"""

import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image

root = '/home/dewei/Desktop/Denoise/OCT_Volumn/'
output_root = '/home/dewei/Desktop/Denoise/OCT_slice/'

for file in os.listdir(root):
    if file.endswith("structure.nii"):
        x_file = file
    else:
        y_file = file

data_nii = nib.load(root+x_file)
x_volume = np.array(data_nii.dataobj)

#%%
def Center_Crop(img, im_size):
    [r,c] = img.shape
    data_crop = np.zeros([im_size,im_size],dtype=np.uint8)
    left = np.int((c-im_size)/2)
    right = c-left
    top = np.int((r-im_size)/2)
    bottom = r-top
    data_crop = img[top:bottom,left:right]
    return data_crop

def cw90(img):
    [r,c] = img.shape
    opt = np.zeros([c,r])
    for i in range(r):
        vector = np.transpose(img[i,:])
        opt[:,r-i-1] = vector
    return opt

for i in range(2500):
    if i % 5 == 0:
        im = x_volume[:,:,i]
        im_crop = cw90(Center_Crop(im,512))
        img = Image.fromarray(np.uint8(im_crop))
        img.save(output_root+'{}.png'.format(i/5))
        
        


