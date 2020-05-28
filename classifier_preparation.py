#!/usr/bin/env python
#
import sys
import os
import numpy as np
import SimpleITK as sitk
import matplotlib
import matplotlib.pyplot as plt
from  scipy import ndimage
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import plot_confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier


def data_loading_normalized():
    c_path = 'COMMON_images_masks/'
    g_path = 'GROUP_images'

    g_images_list = ['g9_77_image.nii','g10_81_image.nii.gz','g9_79_image.nii']
    c_images_list = ['common_40_image.nii.gz','common_41_image.nii.gz','common_42_image.nii.gz']

    g_images = []
    for i in range(len(g_images_list)):
        im_g = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(g_path, g_images_list[i]),sitk.sitkFloat32))
        im_g = im_g/255.0
        g_images.append(im_g)

    c_images = []
    for i in range(len(c_images_list)):
        im = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(c_path, c_images_list[i]),sitk.sitkFloat32))
        im = im/255.0
        c_images.append(im)

    g_lof = []
    c_lof = []

    lof_g77 = np.zeros((g_images[0]).shape[2]) # vector g77
    lof_g77[266:307]=1
    g_lof.append(lof_g77)

    lof_g81 = np.zeros((g_images[1]).shape[2]) # vector g81
    lof_g81[290:336]=1
    g_lof.append(lof_g81)

    lof_g79 = np.zeros((g_images[2]).shape[2]) # vector g79
    lof_g79[266:312]=1
    g_lof.append(lof_g79)

    lof_c40 = np.zeros((c_images[0]).shape[2]) # vector c40
    lof_c40[284:340]=1
    c_lof.append(lof_c40)

    lof_c41 = np.zeros((c_images[1]).shape[2]) # vector c41
    lof_c41[282:324]=1
    c_lof.append(lof_c41)

    lof_c42 = np.zeros((c_images[2]).shape[2]) # vector c42
    lof_c42[274:331]=1
    c_lof.append(lof_c42)

    return g_images, c_images, g_lof, c_lof


def train_classifier(im_list, labels_list, features=True, classifier=None):
    """
    Receive a list of images `im_list` and a list of vectors (one per image) with the labels 0 or 1 depending on the sagittal 2D slice
    contains or not the left obturator foramen. Returns the trained classifier.
    Inputs: im_list = list of images
            labels_list = list of the vectors of each images in im_list (0- slice does not contain LOF, 1- contains)
            features = if True, will extract some features from the image and use them as features for the classifier
                       if False, use image values
            classifier = name of the classfier to train: 'knn' = K-Nearest Neighbor
                                                         'dtree' = Decision Tree
                                                         'rf' = Random Forest
    Outputs: trained_classifier
    """
    if features==True:
        X = []
        for i in range(3):
            for j in range(512):
                #Min1 = np.amin(im_list[i][:240,:,j])
                #Max1 = np.amax(im_list[i][:240,:,j])
                Mean1 = np.mean(im_list[i][:240,:,j])
                Power1 = np.power(im_list[i][:240,:,j], 2)
                Energy1 = Power1.sum()

                #Train_im.append(np.array([[Energy1,Max1,Min1,Mean1]]))
                X.append(np.array([[Energy1,Mean1]]))
    else:
        X = []
        for i in range(3):
            for j in range(512):
                X.append((im_list[i])[:240,:,j])

    Xtrain = np.asarray(X)
    Xtrain = Xtrain.reshape(Xtrain.shape[0],Xtrain.shape[1]*Xtrain.shape[2])
    #print(Xtrain.shape)

    Ytrain = np.concatenate(labels_list, axis=0)
    
    if classifier == 'knn':
        c = KNeighborsClassifier(n_neighbors=30,weights='distance')
        c.fit(Xtrain, Ytrain)
    elif classifier == 'dtree':
        c = DecisionTreeClassifier(criterion='entropy', max_depth=5, splitter='best')
        c.fit(Xtrain, Ytrain)
    elif classifier == 'rf':
        c = RandomForestClassifier(n_estimators=100, criterion='entropy')
        c.fit(Xtrain, Ytrain)
    
    return c


def left_obturator_foramen_selection(im, classifier, features, y_test):
    """
    Receive a CT image and the trained classifier. Returns the sagittal slice number with the maximum 
    probability of containing the left obturator foramen.
    Inputs: im = testing image
            classifier = name of the trained model
            features = if True, will extract some features from the image and use them as features for the classifier
                       if False, use image values
            y_test = labels (groundtruth)
    Outputs: y_preds = predicted values 0 or 1 (0 not present, 1 present)
             y_preds_proba = predicted probabilities in range 0-1.
    
    """
    
    if features == True:
        X = []
        for j in range(512):
            Min1 = np.amin(im[:240,:,j])
            Max1 = np.amax(im[:240,:,j])
            Mean1 = np.mean(im[:240,:,j])
            Power1 = np.power(im[:240,:,j], 2)
            Energy1 = Power1.sum()

            #Train_im.append(np.array([[Energy1,Max1,Min1,Mean1]]))
            X.append(np.array([[Energy1,Mean1]]))
    else:
        X = []
        for i in range(512):
        #for j in range(200,400):
            X.append((im)[:240,:,i])
            
    Xtest = np.asarray(X)
    Xtest = Xtest.reshape(Xtest.shape[0],Xtest.shape[1]*Xtest.shape[2])
    #print(Xtest.shape)
    
    y_preds = classifier.predict(Xtest)
    y_preds_proba = classifier.predict_proba(Xtest)
    
    '''
    class_names= ['0','1']
    
    disp = plot_confusion_matrix(classifier, Xtest, y_test,
                             display_labels=class_names,
                             cmap=plt.cm.Blues)
    disp.ax_.set_title('Confusion matrix')

    print('Confusion matrix')
    print(disp.confusion_matrix)
    '''
    
    return y_preds, y_preds_proba
