#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 13:30:14 2020

@author: dewei
"""

import nibabel as nib
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt

def nii_load(dir):
    data_nii = nib.load(dir)
    data = np.array(data_nii.dataobj)
    return data

root = '/home/dewei/Desktop/slc/'
atlas_list = []
for file in os.listdir(root):
    if file.endswith('.nii.gz'):
        atlas_list.append(file)
        if file.startswith('fix_img'):
            fix_img = Image.fromarray(nii_load(root+file))
            fix_img.save(root+'fix_img.tif')
            
atlas_list.sort()

for i in range(len(atlas_list)):
    img = Image.fromarray(nii_load(root+atlas_list[i]))
    #plt.imshow(img,cmap='gray')
    img.save(root+'atlas{}.tif'.format(i))
    
