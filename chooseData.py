#!/usr/bin/python
# -*- coding:utf8 -*-
import numpy as np
from skimage import io
from scipy import misc
import cv2
import scipy.ndimage as sp
import scipy.misc as scm
import scipy.io as sio
import os
import gc
import natsort
import glob
import dicom
import gc
from matplotlib import pyplot as plt
from Pretreatment import read_dicom_series,step1_preprocess_img_slice

def load_img_label(directory, filepattern = "*"):
    if not os.path.exists(directory) or not os.path.isdir(directory):
        raise ValueError("Given directory does not exist or is a file : " + str(directory))
    print '\tRead data', directory
    lstFilesDCM = natsort.natsorted(glob.glob(os.path.join(directory, filepattern)))
    print '\tLength dicom series', len(lstFilesDCM)
    # Get ref file
    # Load dimensions based on the number of rows, columns, and slices (along the Z axis)
    ConstPixelDims = (len(lstFilesDCM),512,512,1)#下次训练时要改成512×512
    # The array is sized based on 'ConstPixelDims'
   # ArrayDicom = np.zeros(ConstPixelDims, dtype=RefDs.pixel_array.dtype)
    ArrayDicom = np.zeros(ConstPixelDims,dtype=np.float16)
    # loop through all the DICOM files
    for filenameDCM in lstFilesDCM:
        # read the file
        #ds = io.imread(filenameDCM)
        ds = cv2.imread(filenameDCM, cv2.IMREAD_GRAYSCALE)
        # store the raw image data
        ArrayDicom[lstFilesDCM.index(filenameDCM),:, :, 0] = ds
    return ArrayDicom

if __name__ == '__main__':
    j=0
    for i in range(0,131):

        path_new_img ='/home/bai/LiangData_Afterchoose/img/'
        path_new_lesion = '/home/bai/LiangData_Afterchoose/lesion/'
        pathliver = '/home/bai/LiangData2/images_bmp_rotate/'+str(i)
        pathleision = '/home/bai/LiangData2/item_seg_rotate/' + str(i)
        if os.path.exists(pathliver):
       # print(pathliver)
       # print(pathleision)
            img = load_img_label(pathliver)
            lesion = load_img_label(pathleision)
            for k in range(img.shape[0]):
                if len(np.where(lesion[k,:,:,0]>0)[0])>0:
                    print(path_new_img+str(j)+'.bmp')
                    misc.imsave(path_new_img+str(j)+'.bmp',img[k,:,:,0])
                    misc.imsave(path_new_lesion + str(j) + '.bmp', lesion[k, :, :,0])
                    j = j+1