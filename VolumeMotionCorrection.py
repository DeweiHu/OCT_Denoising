# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 08:48:54 2020

@author: hudew
"""

import sys
sys.path.insert(0,'C:\\Users\\hudew\\OneDrive\\桌面\\Denoise\\')
import MotionCorrection

import os
import time
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from skimage import io

'''
1.Human data need to be rescaled to [0,255]
2.Pick single frame 
3.Do motion correction and save in form of packages in a tuple 
'''
global FrameNum
FrameNum = 5

def Rescale(img):
    val_max = img.max()
    val_min = img.min()
    val_range = val_max-val_min
    opt = (img-val_min)/val_range*255
    return np.float32(opt)

def PickFrame(volume):
    dim = volume.shape
    opt = np.zeros([int(dim[0]/FrameNum),dim[1],dim[2]],dtype=np.float32)
    for i in range(dim[0]):
        if i % FrameNum == 0:
            opt[int(i/FrameNum),:,:] = volume[i,:,:]
    return opt

volumeroot = 'E:\\human\\'
volumelist = []
for file in os.listdir(volumeroot):
    if file.endswith('.tif'):
        volumelist.append(file)
volumelist.sort()

volume = Rescale(io.imread(volumeroot+volumelist[18]))
data = PickFrame(volume)

#plt.imshow(data[100,:,:],cmap='gray')
del volume

#%% Pack
def load_nii(dir):
    data_nii = nib.load(dir)
    data = np.array(data_nii.dataobj)
    return data

def save_nii(volume,path,filename):
    output = nib.Nifti1Image(volume,np.eye(4))
    nib.save(output,os.path.join(path,filename))

radius = 7
root = 'E:\\Temp\\'
fixedImageFile = root+'fix_img.nii.gz'
movingImageFile = root+'mov_img.nii.gz'
outputImageFile = root+'opt.nii.gz'
pair = ()

t1 = time.time()
dim = data.shape
for i in range(dim[0]):
    if i >= radius and i < dim[0]-radius:
        fix = data[i,:,:] 
        save_nii(fix,root,'fix_img.nii.gz')
        for j in range(radius):
            dist = j+1
            frame_x = data[i-dist,:,:]
            save_nii(frame_x,root,'mov_img.nii.gz')
            MotionCorrection.MotionCorrect(fixedImageFile,movingImageFile,outputImageFile)
            x_pre = np.zeros([1024,512],dtype=np.float32)
            x_pre[:,:500] = load_nii(outputImageFile)
            
            frame_x = data[i+dist,:,:]
            save_nii(frame_x,root,'mov_img.nii.gz')
            MotionCorrection.MotionCorrect(fixedImageFile,movingImageFile,outputImageFile)
            x_post = np.zeros([1024,512],dtype=np.float32)
            x_post[:,:500] = load_nii(outputImageFile)
            
            y = np.zeros([1024,512],dtype=np.float32)
            y[:,:500] = fix
            pair = pair + ((x_pre,y),(x_post,y),)
        pair = pair + ((y,y),)
        
    if i % 20 == 0 :
        print('{} slices have been completed'.format(i))

t2 = time.time()
print('Time:{} min'.format((t2-t1)/60))

#%% Save the aligned data
import pickle
with open('E:\\VoxelMorph\\test_pair.pickle','wb') as f:
    pickle.dump(pair,f)
    
    
    