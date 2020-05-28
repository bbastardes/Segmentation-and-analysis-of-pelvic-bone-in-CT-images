#!/usr/bin/env python
#
import sys
import os
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
from  scipy import ndimage


def assessment(mask_ref, majority_cl):
    '''
    Calculates the similarity of two images using dice coef and hausdorff distance
    as metrics.
    '''
    
    #gt1 = sitk.Cast(mask_ref, sitk.sitkUInt8) == 2 # 1- RF, 2- LF, 3- RH, 4- LH, 5- S
    #gt2 = sitk.Cast(mask_ref, sitk.sitkUInt8) == 4
    #gt = gt1+gt2

    #convert groundtruth and mask to same properties and space
    gt = sitk.Cast(mask_ref, sitk.sitkUInt8) #== label
    my_mask = sitk.Cast(majority_cl, sitk.sitkUInt8) 

    my_mask.SetOrigin(gt.GetOrigin())
    my_mask.SetSpacing(gt.GetSpacing())

    #dice coeff
    dice_dist = sitk.LabelOverlapMeasuresImageFilter()
    dice_dist.Execute(gt, my_mask)
    dice = dice_dist.GetDiceCoefficient() 
    #print('Dice: '+str(dice))

    #hausdorff dist
    hausdorff_dist = sitk.HausdorffDistanceImageFilter()
    hausdorff_dist.Execute(gt, my_mask)
    hausdorff = hausdorff_dist.GetHausdorffDistance() 
    #print('Hausdorff: '+str(hausdorff))
    
    #for i in range(1,500,25):
        #plt.imshow(sitk.GetArrayFromImage(gt-my_mask)[:,i,:])
        #plt.show()
    return dice, hausdorff



def compare_bones(c_masks, c_manual_masks):
    '''
    calls assessment function to compare by bones the similarity of the two images
    '''
    
    # right femur
    dice, hausdorff = assessment(c_masks==1, c_manual_masks==1)
    print('Right Femur -> Dice: '+str(dice) +' Hausdorff: '+str(hausdorff))
    # left femur
    dice, hausdorff = assessment(c_masks==2, c_manual_masks==5)
    print('Left Femur  -> Dice: '+str(dice) +' Hausdorff: '+str(hausdorff))
    # right hip
    dice, hausdorff = assessment(c_masks==3, c_manual_masks==2)
    print('Right Hip   -> Dice: '+str(dice) +' Hausdorff: '+str(hausdorff))
    # left hip
    dice, hausdorff = assessment(c_masks==4, c_manual_masks==4)
    print('Left Hip    -> Dice: '+str(dice) +' Hausdorff: '+str(hausdorff))
    return

