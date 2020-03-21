#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 10:52:37 2020

@author: dewei
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
import os
from PIL import Image

def nii_save(volume,path,filename):
    output = nib.Nifti1Image(volume,np.eye(4))
    nib.save(output,os.path.join(path,filename))
    
root = '/home/dewei/Desktop/slc/'
file = 'warp.pickle'

# registered image tif
with open(root+file,'rb') as f:
    warp = pickle.load(f)
for i in range(15):
    im = Image.fromarray(warp[i,:,:])
    im.save(root+'atlas{}.tif'.format(i))

nii_save(warp,root,'warp.nii.gz')

# fix image tif
with open(root+'y.pickle','rb') as f:
    fix = pickle.load(f)
fix_img = Image.fromarray(fix[0,:,:])
fix_img.save(root+'fix_img.tif')

nii_save(fix,root,'fix.nii.gz')

# moving image tif
with open(root+'x.pickle','rb') as f:
    mov = pickle.load(f)

nii_save(mov,root,'mov_img.nii.gz')

#%%
from skimage import io
im_smooth = io.imread(root+'synthResult.tif')




