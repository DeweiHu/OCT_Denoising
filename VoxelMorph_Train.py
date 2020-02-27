#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 15:41:04 2020

@author: hud4
"""

import sys
sys.path.insert(0,'/home/hud4/Desktop/2020/VoxelMorph/')

import numpy as np
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal

import losses


device = torch.device("cuda:0" if( torch.cuda.is_available() and gpu>0 ) else "cpu")


#%%
class conv_block(nn.Module):
    """
    [Input] (dim, in_channels, out_channels, stride)
    [Output] single convolution block
    """
    def __init__(self, dim, in_channels, out_channels, stride):
        super(conv_block, self).__init__()
        
        conv_fn = getattr(nn,"Conv{0}d".format(dim))
        
        if stride == 1:
            ksize = 3
        elif stride == 2:
            ksize = 4
        else:
            raise Exception('stride must be 1 or 2')
        
        self.main = conv_fn(in_channels, out_channels, ksize, stride, 1)
        self.activation = nn.LeakyReLU(0.2)
        
    def forward(self, x):
        x = self.main(x)
        out = self.activation(x)
        return out

class unet_core(nn.Module):
    """
    [Input] a fixed image and a moving image
    [Output] a flow-field
    """
    def __init__(self, dim, enc_nf, dec_nf, full_size=True):
        """
        Initiate U-Net
        param dim:       dimension of the image passed into the network
        param enc_nf:    number of feature maps in each layer of encoding stage
        param dec_nf:    number of feature maps in each layer of decoding stage
        param full_size: boolean value representing whether full amount of decoding layers
        """
        super(unet_core, self).__init__()
        
        self.full_size = full_size
        self.vm2 = len(dec_nf) == 7
        
        # Encoder functions
        self.enc = nn.ModuleList()
        for i in range(len(enc_nf)):
            if i == 0:
                prev_nf = 2
            else:
                prev_nf = enc_nf[i-1]
            self.enc.append(conv_block(dim, prev_nf, enc_nf[i], 2))
        
        # Decoder functions
        self.dec = nn.ModuleList()
        self.dec.append(conv_block(dim, enc_nf[-1], dec_nf[0]))
        self.dec.append(conv_block(dim, dec_nf[0]*2, dec_nf[1]))
        self.dec.append(conv_block(dim, dec_nf[1]*2, dec_nf[2]))
        self.dec.append(conv_block(dim, dec_nf[2]+enc_nf[0], dec_nf[3]))
        self.dec.append(conv_block(dim, dec_nf[3], dec_nf[4]))
        
        if self.full_size:
            self.dec.append(conv_block(dim, dec_nf[4]+2, dec_nf[5], 1))
        
        if self.vm2:
            self.vm2_conv = conv_block(dim, dec_nf[5], dec_nf[6])
        
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        
    def forward(self, x):
        """
        [Input] concatenate fixed and moving image 
        """
        # A good way to iteratively do conv(downsample) and save mid results
        x_enc = [x]
        for func in self.enc:
            x_enc.append(func(x_enc[-1]))
        
        # Three conv + upsample + concatenate series
        y = x_enc[-1]
        for i in range(3):
            y = self.dec[i](y)
            y = self.upsample(y)
            y = torch.cat([y,x_enc[-(i+2)]],dim=1)
        
        # Two convs at full_size/2 res
        y = self.dec[3](y)
        y = self.dec[4](y)
        
        # upsample to full res, concatenate and conv
        if self.full_size:
            y = self.upsample(y)
            y = torch.cat([y,x_enc[0]],dim=1)
            y = self.dec[5](y)
        
        if self.vm2:
            y = self.vm2_conv(y)
            
        return y
        
class SpatialTransformer(nn.Module):
    """
    [Input] Deformation field predicted by the U-Net
    [Output] Transformed image
    param size: size of the input to the spatial transformer block
    param mode: method of interpolation for grid_sampler
    """
    
    def __init__(self, size, mode='bilinear'):
        super(SpatialTransformer, self).__init__()
        
        # Create sampling grid
        """
        Grid represent the original coordinate (identity tranformation)
        """
        vectors = [ torch.arange(0,s) for s in size ]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0) # add batch number
        grid = grid.type(torch.FloatTensor)
        self.register_buffer('grid', grid)
        
        self.mode = mode
        
    def forward(self, src, flow):
        """
        param src: moving image
        param flow: deformation field
        """
        new_locs = self.grid+flow
        
        shape = flow.shape[2:]
        
        # Need to normalize grid values to [-1,1] for resampler
        for i in range(len(shape)):
            new_locs[:,i,...] = 2*(new_locs[:,i,...]/(shape[i]-1)-0.5)
        
        if len(shape) == 2:
            new_locs = new_locs.permute(0,2,3,1)
            new_locs = new_locs[...,[1,0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0,2,3,4,1)
            new_locs = new_locs[...,[2,1,0]]
        
        return F.grid_sample(src, new_locs, mode=self.mode)

class cvpr2018_net(nn.Module):
    
    def __init__(self, vol_size, enc_nf, dec_nf, full_size=True):
        super(cvpr2018_net, self).__init__()
        
        dim = len(vol_size)
        self.unet_model = unet_core(dim, enc_nf, dec_nf, full_size)
        
        conv_fn = getattr(nn,'Conv%dd' % dim)
        self.flow = conv_fn(dec_nf[-1], dim, kernel_size=3, padding=1)
        
        # Make flow weights + bias small
        nd = Normal(0,1e-5)
        self.flow.weight = nn.Parameter(nd.sample(self.flow.weight.shape))
        self.flow.bias = nn.Parameter(torch.zeros(self.flow.bias.shape))
        
        self.spatial_transform = SpatialTransformer(vol_size)
        
    def forward(self, src, tgt):
        x = torch.cat([src,tgt],dim=1)
        x = self.unet_model(x)
        flow = self.flow(x)
        y = self.spatial_transform(src, flow)
        
        return y, flow

#%%
enc_nf = [16,32,32,32]
dec_nf = [32,32,32,32,8,8]

model = cvpr2018_net(vol_size, enc_nf, dec_nf)
model.to(device)

# Set optimizer
from torch.optim import Adam
opt = Adam(model.parameters(), lr=1e-4)

sim_loss_fn = losses.mse_loss
grad_loss_fn = losses.gradient_loss
        
        
        
            

