# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 15:35:51 2020

@author: hudew
"""

import sys
sys.path.insert(0,'C:\\Users\\hudew\\OneDrive\\桌面\\Denoise\\')
import MyFunctions

import os
import numpy as np
import matplotlib.pyplot as plt

root = 'E:\\denoise result\\'
volumelist = []

for volume in os.listdir(root):
    volumelist.append(volume)

#%% Fovea grand image creation
Fovea_y = MyFunctions.nii_loader(root+volumelist[0]+'\\'+volumelist[0]+'_aver.nii.gz')
Fovea_x = MyFunctions.nii_loader(root+volumelist[0]+'\\'+volumelist[0]+'_noi.nii.gz')
Fovea_sf = MyFunctions.nii_loader(root+volumelist[0]+'\\'+volumelist[0]+'_sf.nii.gz')
Fovea_dn = MyFunctions.nii_loader(root+volumelist[0]+'\\'+volumelist[0]+'_dn.nii.gz')

slc = [335,437,200,308]

Fovea_grand = np.zeros([512*4-12,512*4-12],dtype=np.float32)
for i in range(4):
    Fovea_grand[i*512:i*512+500,0:500] = Fovea_x[slc[i],:500,:]
    Fovea_grand[i*512:i*512+500,512:1012] = Fovea_sf[slc[i],:500,:]
    Fovea_grand[i*512:i*512+500,1024:1524] = Fovea_dn[slc[i],:500,:]
    Fovea_grand[i*512:i*512+500,1536:2036] = Fovea_y[slc[i],:500,:]

plt.figure(figsize=(21,21))
plt.axis('off')
plt.imshow(Fovea_grand,cmap='gray')
plt.savefig(root+'Fovea_SNR_96.png')   
plt.show()

#%% ONH grand image creation
ONH_y = MyFunctions.nii_loader(root+volumelist[2]+'\\'+volumelist[2]+'_aver.nii.gz')
ONH_x = MyFunctions.nii_loader(root+volumelist[2]+'\\'+volumelist[2]+'_noi.nii.gz')
ONH_sf = MyFunctions.nii_loader(root+volumelist[2]+'\\'+volumelist[2]+'_sf.nii.gz')
ONH_dn = MyFunctions.nii_loader(root+volumelist[2]+'\\'+volumelist[2]+'_dn.nii.gz')

slc = [244,205,222,337]

ONH_grand = np.zeros([512*4-12,512*4-12],dtype=np.float32)
for i in range(4):
    ONH_grand[i*512:i*512+500,0:500] = ONH_x[slc[i],100:600,:]
    ONH_grand[i*512:i*512+500,512:1012] = ONH_sf[slc[i],100:600,:]
    ONH_grand[i*512:i*512+500,1024:1524] = ONH_dn[slc[i],100:600,:]
    ONH_grand[i*512:i*512+500,1536:2036] = ONH_y[slc[i],100:600,:]

plt.figure(figsize=(21,21))
plt.axis('off')
plt.imshow(ONH_grand,cmap='gray')
plt.savefig(root+'ONH_SNR_96.png')   
plt.show()