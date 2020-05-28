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

def data_loading_all():
  ''' Loading of images (commmon and group) and their segmentations)'''
  c_path = 'COMMON_images_masks/'
  c_images_list = ['common_40_image.nii.gz','common_41_image.nii.gz','common_42_image.nii.gz']
  c_masks_list = ['common_40_mask.nii.gz','common_41_mask.nii.gz','common_42_mask.nii.gz']
  g_path = 'GROUP_images'
  g_images_list = ['g9_77_image.nii','g10_81_image.nii.gz','g9_79_image.nii']
  g_masks_list = ['g9_77_image_mask_l.nii.gz','g9_77_image_mask_r.nii.gz','g9_77_image_mask_p.nii.gz','g10_81_image_mask.nii.gz','g9_79_image_mask.nii']

  c_images = []
  c_masks = []
  for i in range(len(c_images_list)):
      im = sitk.ReadImage(os.path.join(c_path, c_images_list[i]),sitk.sitkFloat32)
      c_images.append(im)
      mask = sitk.ReadImage(os.path.join(c_path, c_masks_list[i]),sitk.sitkUInt8)
      c_masks.append(mask)
      
  g_images = []
  g_masks = []
  for i in range(len(g_images_list)):
      im_g = sitk.ReadImage(os.path.join(g_path, g_images_list[i]),sitk.sitkFloat32)
      g_images.append(im_g)
  for i in range(len(g_masks_list)):
      mask = sitk.ReadImage(os.path.join(g_path, g_masks_list[i]),sitk.sitkUInt8)
      g_masks.append(mask)

  return c_images, c_masks, g_images, g_masks

def masks_per_parts(c_images, c_masks, g_images, g_masks):
  ''' This function divides the masks into left and right part for both groups of images
      order of provided masks = 1RF 2LF 3RH 4LH 5Sacrum
      order of my masks = 1RF 2RH 3Sacrum 4LH 5LF
      *exception in g_masks0 = 1LF 2LH
                    g_masks1 = 1RF 2RH
                    g_masks2 = Sacrum
'''

  c_masks_left = []
  c_masks_left.append((c_masks[0]==2) + 2*(c_masks[0]==4))
  c_masks_left.append((c_masks[1]==2) + 2*(c_masks[1]==4))
  c_masks_left.append((c_masks[2]==2) + 2*(c_masks[2]==4))

  g_masks_left = []
  g_masks_left.append((g_masks[0]==2) + 2*(g_masks[0]==1))   # left femur & hip mask image 77
  g_masks_left.append((g_masks[3]==5) + 2*(g_masks[3]==4))   # left femur & hip mask image 81
  g_masks_left.append((g_masks[4]==5) + 2*(g_masks[4]==4))   # left femur & hip mask image 79

  c_masks_right = []
  c_masks_right.append((c_masks[0]==1) + 2*(c_masks[0]==3))
  c_masks_right.append((c_masks[1]==1) + 2*(c_masks[1]==3))
  c_masks_right.append((c_masks[2]==1) + 2*(c_masks[2]==3))

  g_masks_right = []
  g_masks_right.append((g_masks[1]==1) + 2*(g_masks[1]==2))   # right femur & hip mask image 77
  g_masks_right.append((g_masks[3]==1) + 2*(g_masks[3]==2))   # right femur & hip mask image 81
  g_masks_right.append((g_masks[4]==1) + 2*(g_masks[4]==2))   # right femur & hip mask image 79

  return c_masks_left, g_masks_left, c_masks_right, g_masks_right

def get_common_manual_masks():
  c_path = '/COMMON_masks_manual/'
  c_manual_masks_list = ['common_40_image_mask.nii.gz','common_41_image_mask.nii.gz','common_42_image_mask_l.nii.gz', 'common_42_image_mask_r.nii.gz']

  c_manual_masks = []
  for i in range(len(c_manual_masks_list)):
      mask = sitk.ReadImage(os.path.join(c_path, c_manual_masks_list[i]),sitk.sitkUInt8)
      c_manual_masks.append(mask)

  # order of the labels in 40 and 41: 1RF 2RH 3Sacrum 4LH 5LF
  # mix last two to create a unique one for 42
  c_masks = []
  c_masks.append(c_manual_masks[0])
  c_masks.append(c_manual_masks[1])
  c_masks.append(c_manual_masks[3]+(c_manual_masks[2]==1)*5 + (c_manual_masks[2]==2)*4)
  

  return c_masks
 
