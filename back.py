# !/usr/bin/python
# -*- coding:utf8 -*-
import numpy as np
from skimage import io

import scipy.ndimage as sp
import scipy.misc as misc
import scipy.io as sio
import os
import gc
import natsort
import glob
import dicom
import gc
from matplotlib import pyplot as plt
from Pretreatment import read_dicom_series, step1_preprocess_img_slice
from loadData import load_img_label,load_mask,Cropped_fill
import tflearn
def return2img():

    n = 5

    img_prep = tflearn.ImagePreprocessing()
    img_prep.add_featurewise_zero_center(per_channel=True)

    # Real-time data augmentation
    img_aug = tflearn.ImageAugmentation()
    img_aug.add_random_flip_leftright()
    img_aug.add_random_crop([32, 32], padding=4)

    # Building Residual Network
    net = tflearn.input_data(shape=[None, 32, 32, 1],
                             data_preprocessing=img_prep,
                             data_augmentation=img_aug)
    net = tflearn.conv_2d(net, 16, 3, regularizer='L2', weight_decay=0.0001)
    net = tflearn.residual_block(net, n, 16)
    net = tflearn.residual_block(net, 1, 32, downsample=True)
    net = tflearn.residual_block(net, n - 1, 32)
    net = tflearn.residual_block(net, 1, 64, downsample=True)
    net = tflearn.residual_block(net, n - 1, 64)
    net = tflearn.batch_normalization(net)
    net = tflearn.activation(net, 'relu')
    net = tflearn.global_avg_pool(net)
    # Regression
    net = tflearn.fully_connected(net, 3, activation='softmax')
    mom = tflearn.Momentum(0.1, lr_decay=0.1, decay_step=32000, staircase=True)
    net = tflearn.regression(net, optimizer=mom,
                             loss='categorical_crossentropy')
    # Training
    model = tflearn.DNN(net)
    model.load("model_resnet_+-------cifar10-6500")
    img = load_img_label('/media/bai/Elements/LiangData_Afterchoose/img1')
    livermask = load_mask('/media/bai/Elements/LiangData_Afterchoose/mask1')
    abc=[]
    np.array(abc)
    for i in range(img.shape[0]):
        a = np.zeros((512,512,3,3),dtype=np.float32)#最后一项表示第几个分割方式，倒数第二项表示预测的三个结果
        for j in range(0,3):
           this_mask =  livermask[i, :, :, j]
           this_mask_num = int(np.max(this_mask))
           for k in range(1,this_mask_num+1):
               if os.path.exists('/media/bai/Elements/LiangData_Afterchoose/superpixel/'+str(i)+'_'+str(j)+'_'+str(k)+'.jpg'):
                   this_img = io.imread('/media/bai/Elements/LiangData_Afterchoose/superpixel/'+str(i)+'_'+str(j)+'_'+str(k)+'.jpg')
                   this_img = np.reshape(this_img,(1,32,32,1))
                   this_img = this_img.astype(np.float32)
                   result = model.predict(this_img)
                  # print(result)
                   abc.append(result)
                   thisPartLoc=np.where(this_mask==k)
                   for num in range(len(thisPartLoc[1])):
                       a[thisPartLoc[0][num],thisPartLoc[1][num],0,j] = result[0,0]
                       a[thisPartLoc[0][num],thisPartLoc[1][num], 1, j] = result[0,1]
                       a[thisPartLoc[0][num],thisPartLoc[1][num], 2, j] = result[0,2]
        b = np.max(a,axis=3)
        final = np.argmax(b,axis=2)
        #添加3D CRF
        misc.imsave('/media/bai/Elements/LiangData_Afterchoose/result/' + str(i) + '.jpg', final)
if __name__ == '__main__':
    return2img()