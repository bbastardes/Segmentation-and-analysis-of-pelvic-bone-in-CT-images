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

def est_lin_transf(im_ref, im_mov, mask=None):
    """
    Estimate linear transform to align `im_mov` to `im_ref` and return the transform parameters.
    """ 
    
    im_ref.SetOrigin([0,0,0])
    im_mov.SetOrigin([0,0,0])    
    
    elastixImageFilter = sitk.ElastixImageFilter()
    elastixImageFilter.SetFixedImage(im_ref)
    elastixImageFilter.SetMovingImage(im_mov)

    parameterMap = sitk.GetDefaultParameterMap('affine')
    #parameterMap['MaximumNumberOfIterations'] = ['1000']
    parameterMap['Interpolator'] = ["BSplineInterpolator"]
    
    if mask:
        mask.SetOrigin(im_ref.GetOrigin())
        mask.SetSpacing(im_ref.GetSpacing())
        elastixImageFilter.SetFixedMask(mask)
        parameterMap['ImageSampler'] = ['RandomSparseMask'] #if mask is too small

    elastixImageFilter.SetParameterMap(parameterMap)
    #elastixImageFilter.LogToConsoleOn()
    elastixImageFilter.Execute()
    transform = elastixImageFilter.GetTransformParameterMap()
    
    return transform


def apply_lin_transf(im_mov, lin_xfm):
    """
    Apply given linear transform `lin_xfm` to `im_mov` and return the transformed image.
    """
    im_mov.SetOrigin([0,0,0])
    transformixImageFilter = sitk.TransformixImageFilter()
    lin_xfm[0]['ResampleInterpolator'] = ['FinalNearestNeighborInterpolator']
    transformixImageFilter.SetTransformParameterMap(lin_xfm)
    transformixImageFilter.SetMovingImage(im_mov)
    transformixImageFilter.Execute()    
    transformed_im = transformixImageFilter.GetResultImage()
    
    return transformed_im


def est_nl_transf(im_ref, im_mov, fixed_mask, mov_mask):
    """
    Estimate non-linear transform to align `im_mov` to `im_ref` and return the transform parameters.
    fixed_mask = mask to apply to the reference image
    mov_mask = mask to apply to the moving image
    
    """
    
    im_ref.SetOrigin([0,0,0])
    im_mov.SetOrigin([0,0,0])
    
    elastixImageFilter = sitk.ElastixImageFilter()
    elastixImageFilter.SetFixedImage(im_ref)
    elastixImageFilter.SetMovingImage(im_mov)
    
    parameterMapVector = sitk.GetDefaultParameterMap("bspline")
    parameterMapVector['MaximumNumberOfIterations'] = ['1000']
    
    if fixed_mask:
        fixed_mask.SetOrigin(im_ref.GetOrigin())
        fixed_mask.SetSpacing(im_ref.GetSpacing())
        elastixImageFilter.SetFixedMask(fixed_mask)
        parameterMapVector['ImageSampler'] = ['RandomSparseMask'] #if mask is too small
    
    if mov_mask:
        mov_mask.SetOrigin(im_mov.GetOrigin())
        mov_mask.SetSpacing(im_mov.GetSpacing())
        elastixImageFilter.SetMovingMask(mov_mask)
        parameterMapVector['ImageSampler'] = ['RandomSparseMask'] #if mask is too small

    
    elastixImageFilter.LogToConsoleOn()
    elastixImageFilter.Execute()

    transform = elastixImageFilter.GetTransformParameterMap()
    
    return transform


def apply_nl_transf(im_mov, nl_xfm):
    """
    Apply given non-linear transform `nl_xfm` to `im_mov` and return the transformed image.
    """
    im_mov.SetOrigin([0,0,0])
    transformixImageFilter = sitk.TransformixImageFilter()
    nl_xfm[0]['ResampleInterpolator'] = ['FinalNearestNeighborInterpolator']
    transformixImageFilter.SetTransformParameterMap(nl_xfm)
    transformixImageFilter.SetMovingImage(im_mov)
    transformixImageFilter.Execute()    
    transformed_im = transformixImageFilter.GetResultImage()
    return transformed_im
