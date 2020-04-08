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
import matplotlib

# Adjust the fontsize
font_size = 15
matplotlib.rc('font', size=font_size)
matplotlib.rc('axes', titlesize=font_size)

root = 'E:\\denoise result\\'
volumelist = []

for volume in os.listdir(root):
    if volume.endswith('_2'):
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

#%% Evaluations 
slc = 308
seg1 = 355
seg2 = 300

# verticle segment visualization
img_x = Fovea_x[slc,100:400,:]
img_y = Fovea_y[slc,100:400,:]
img_sf = Fovea_sf[slc,100:400,:]
img_dn = Fovea_dn[slc,100:400,:]

plt.figure(figsize=(12,8))
ax1 = plt.subplot2grid((2,2),(0,0),colspan=1,rowspan=1)
ax1.set_title('noisy')
ax1.imshow(img_x,cmap='gray')
ax1.axvline(x = seg1, linewidth=1.5, color='r', linestyle='--')
ax1.axvline(x = seg2, linewidth=1.5, color='b', linestyle='--')

ax2 = plt.subplot2grid((2,2),(0,1),colspan=1,rowspan=1)
ax2.set_title('5-averge')
ax2.imshow(img_y,cmap='gray')
ax2.axvline(x = seg1, linewidth=1.5, color='r', linestyle='--')
ax2.axvline(x = seg2, linewidth=1.5, color='b', linestyle='--')

ax3 = plt.subplot2grid((2,2),(1,0),colspan=1,rowspan=1)
ax3.set_title('self-fusion')
ax3.imshow(img_sf,cmap='gray')
ax3.axvline(x = seg1, linewidth=1.5, color='r', linestyle='--')
ax3.axvline(x = seg2, linewidth=1.5, color='b', linestyle='--')

ax4 = plt.subplot2grid((2,2),(1,1),colspan=1,rowspan=1)
ax4.set_title('denoise')
ax4.imshow(img_dn,cmap='gray')
ax4.axvline(x = seg1, linewidth=1.5, color='r', linestyle='--')
ax4.axvline(x = seg2, linewidth=1.5, color='b', linestyle='--')

#%% seg 1: vessel

vec_x = img_x[:,seg1]
vec_y = img_y[:,seg1]
vec_sf = img_sf[:,seg1]
vec_dn = img_dn[:,seg1]

plt.figure(figsize=(15,6))
plt.title('Red cross-section, denoise vs average')
plt.ylabel('Intensity value')
plt.xlabel('Depth of retina')
plt.plot(np.arange(len(vec_y)),vec_x,color='silver',linewidth=1.5)
plt.plot(np.arange(len(vec_y)),vec_dn,color='navy',linewidth=1.5)
plt.plot(np.arange(len(vec_y)),vec_y,color='r',linewidth=2)
plt.axvline(x = 57, color='black', linestyle = '--')
plt.axvline(x = 122, color='black', linestyle = '--')

plt.figure(figsize=(15,6))
plt.title('Red cross-section, self-fusion vs average')
plt.ylabel('Intensity value')
plt.xlabel('Depth of retina')
plt.plot(np.arange(len(vec_y)),vec_sf,color='skyblue',linewidth=1.5)
plt.plot(np.arange(len(vec_y)),vec_y,color='r',linewidth=2)
plt.axvline(x = 57, color='black', linestyle = '--')
plt.axvline(x = 122, color='black', linestyle = '--')

#%% seg 2: layers

vec_x = img_x[:,seg2]
vec_y = img_y[:,seg2]
vec_sf = img_sf[:,seg2]
vec_dn = img_dn[:,seg2]

plt.figure(figsize=(15,6))
plt.title('Blue cross-section, denoise vs average')
plt.ylabel('Intensity value')
plt.xlabel('Depth of retina')
plt.plot(np.arange(len(vec_y)),vec_x,color='silver',linewidth=1.5)
plt.plot(np.arange(len(vec_y)),vec_dn,color='navy',linewidth=1.5)
plt.plot(np.arange(len(vec_y)),vec_y,color='r',linewidth=2)
plt.axvline(x = 70, color='black', linestyle = '--')
plt.axvline(x = 90, color='black', linestyle = '--')
plt.axvline(x = 227, color='black', linestyle = '--')
plt.axvline(x = 260, color='black', linestyle = '--')

plt.figure(figsize=(15,6))
plt.title('Blue cross-section, self-fusion vs average')
plt.ylabel('Intensity value')
plt.xlabel('Depth of retina')
plt.plot(np.arange(len(vec_y)),vec_sf,color='skyblue',linewidth=1.5)
plt.plot(np.arange(len(vec_y)),vec_y,color='r',linewidth=2)
#plt.axvline(x = 57, color='black', linestyle = '--')
#plt.axvline(x = 122, color='black', linestyle = '--')