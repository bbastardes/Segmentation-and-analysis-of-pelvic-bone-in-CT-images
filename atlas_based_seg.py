#!/usr/bin/env python
#
import sys
import os
import numpy as np
import SimpleITK as sitk
#from PIL import Image
#from pylab import *
import matplotlib.pyplot as plt
from  scipy import ndimage
from registration import *

def majority_v(im_ref,masks):
    # Majority Voting
    labelForUndecidedPixels = 0
    majority_voting = sitk.LabelVoting(masks, labelForUndecidedPixels)
    majority_voting.SetOrigin(im_ref.GetOrigin())

    # Apply morphological operation of Closing to close holes
    closingFilter = sitk.BinaryMorphologicalClosingImageFilter()
    closingFilter.SetKernelType(sitk.BinaryMorphologicalClosingImageFilter.Ball)
    closingFilter.SetKernelRadius(10)
    majority_closed = closingFilter.Execute(majority_voting)
    return majority_closed

def seg_atlas(im_ref, atlas_ct_list, atlas_seg_list): 
    """
    Apply atlas-based segmentation of `im` using the list of CT images in `atlas_ct_list` and the corresponding
    segmentation masks in `atlas_seg_list`. Return the resulting segmentation mask after majority voting.
    
    im: reference image (from common)
    atlas_ct_list: list of images to register to im (group ones)
    atlas_seg_list: list of the previous images segmented  
    """
    
    reg_masks = []
    for i in range(len(atlas_ct_list)):
    
        im_mov = atlas_ct_list[i]
        mask_mov = atlas_seg_list[i]
    
        # Linear
        lin_transform = est_lin_transf(im_ref, im_mov)
        print('Linear estimation done')
        mov_img_resampled = apply_lin_transf(im_mov, lin_transform)
        lin_reg_mask = apply_lin_transf(mask_mov, lin_transform)
        print('L registration done')
    
        # middle step to create masks to focus when non-L: thresholding based on hounsfield values.
        refmask = im_ref>400
        dilationFilter = sitk.BinaryDilateImageFilter()
        dilationFilter.SetKernelRadius(35)
        fixed_mask = dilationFilter.Execute(refmask,0,1,False)
        
        movmask = mov_img_resampled>400
        #movmask = (lin_reg_mask==1) + (lin_reg_mask==2)
        dilationFilter = sitk.BinaryDilateImageFilter()
        dilationFilter.SetKernelRadius(35)
        mov_mask = dilationFilter.Execute(movmask,0,1,False)

        # NonLinear
        nl_transform = est_nl_transf(im_ref, mov_img_resampled, fixed_mask, mov_mask)
        print('Non-linear estimation done')
        nl_reg_im = apply_nl_transf(mov_img_resampled,nl_transform)
        nl_reg_mask = apply_nl_transf(lin_reg_mask,nl_transform)
        print('NL registration done')
        
        reg_masks.append(sitk.Cast(nl_reg_mask, sitk.sitkUInt8))
    
        print('-------plots coming-------')
        
        r = sitk.GetArrayFromImage(im_ref)
        m = sitk.GetArrayFromImage(im_mov)
        im_r_l = sitk.GetArrayFromImage(mov_img_resampled)
        im_r_nl = sitk.GetArrayFromImage(nl_reg_im)

        mask = sitk.GetArrayFromImage(mask_mov)
        mask_r_l = sitk.GetArrayFromImage(lin_reg_mask)
        mask_r_nl = sitk.GetArrayFromImage(nl_reg_mask)
        
        plt.figure(figsize=(30,60))
        plt.subplot(151), plt.imshow(np.flipud(r[:,250,:])), plt.title('ref') 
        plt.subplot(152), plt.imshow(np.flipud(m[:,250,:])), plt.title('mov') 
        plt.subplot(153), plt.imshow(np.flipud(im_r_l[:,250,:])), plt.title('im_l_registered') 
        plt.subplot(154), plt.imshow(np.flipud(im_r_nl[:,250,:])), plt.title('im_nl_registered') 
        plt.subplot(155), plt.imshow(np.flipud(r[:,250,:]), cmap='Blues'), plt.title('ref - im_nl_registered') 
        plt.imshow(np.flipud(im_r_nl[:,250,:]), cmap='Reds', alpha=0.3) # mask

        plt.figure(figsize=(30,60))
        plt.subplot(151), plt.imshow(np.flipud(r[:,250,:])), plt.title('ref')
        plt.subplot(152), plt.imshow(np.flipud(mask[:,250,:])), plt.title('mask')
        plt.subplot(153), plt.imshow(np.flipud(mask_r_l[:,250,:])), plt.title('mask_l_registered')
        plt.subplot(154), plt.imshow(np.flipud(mask_r_nl[:,250,:])), plt.title('mask_nl_registered')
        plt.subplot(155), plt.imshow(np.flipud(r[:,250,:]), cmap='Blues'), plt.title('ref - mask_nl_registered')
        plt.imshow(np.flipud(mask_r_nl[:,250,:]), cmap='Reds', alpha=0.3) # mask
        plt.show()
    
        print('------plots done!------')

    femur=[]
    hip=[]
    for i in range(3):
        femur.append(reg_masks[i]==1) 
        hip.append(reg_masks[i]==2)
        
    majority_v_femur = majority_voting(im_ref,femur)
    majority_v_hip = majority_voting(im_ref,hip)

    '''    
    # Majority Voting
    labelForUndecidedPixels = 0
    majority_voting = sitk.LabelVoting(reg_masks, labelForUndecidedPixels)
    majority_voting.SetOrigin(im_ref.GetOrigin())

    # Apply morphological operation of Closing to close holes
    closingFilter = sitk.BinaryMorphologicalClosingImageFilter()
    closingFilter.SetKernelType(sitk.BinaryMorphologicalClosingImageFilter.Ball)
    closingFilter.SetKernelRadius(10)
    majority_closed = closingFilter.Execute(majority_voting)
    '''
    return majority_v_femur, majority_v_hip
