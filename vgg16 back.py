# !/usr/bin/python
# -*- coding:utf8 -*-
import numpy as np
from skimage import io
from scipy import misc
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
from Pretreatment import read_dicom_series, step1_preprocess_img_slice
from loadData import load_img_label,load_mask,Cropped_fill
import tflearn
def return2img():

    n = 5
    # Real-time data preprocessing
    img_prep = tflearn.ImagePreprocessing()
    img_prep.add_featurewise_zero_center(per_channel=True)

    # Real-time data augmentation
    img_aug = tflearn.ImageAugmentation()
    img_aug.add_random_flip_leftright()
    img_aug.add_random_crop([32, 32], padding=4)
    inp = tflearn.input_data(shape=[None, 32, 32, 1], data_preprocessing=img_prep, data_augmentation=img_aug,
                             name='input')

    conv1_1 = tflearn.conv_2d(inp, 64, 3, activation='relu', name="conv1_1")
    conv1_2 = tflearn.conv_2d(conv1_1, 64, 3, activation='relu', name="conv1_2")
    pool1 = tflearn.max_pool_2d(conv1_2, 2, strides=2)

    conv2_1 = tflearn.conv_2d(pool1, 128, 3, activation='relu', name="conv2_1")
    conv2_2 = tflearn.conv_2d(conv2_1, 128, 3, activation='relu', name="conv2_2")
    pool2 = tflearn.max_pool_2d(conv2_2, 2, strides=2)

    conv3_1 = tflearn.conv_2d(pool2, 256, 3, activation='relu', name="conv3_1")
    conv3_2 = tflearn.conv_2d(conv3_1, 256, 3, activation='relu', name="conv3_2")
    conv3_3 = tflearn.conv_2d(conv3_2, 256, 3, activation='relu', name="conv3_3")
    pool3 = tflearn.max_pool_2d(conv3_3, 2, strides=2)

    conv4_1 = tflearn.conv_2d(pool3, 512, 3, activation='relu', name="conv4_1")
    conv4_2 = tflearn.conv_2d(conv4_1, 512, 3, activation='relu', name="conv4_2")
    conv4_3 = tflearn.conv_2d(conv4_2, 512, 3, activation='relu', name="conv4_3")
    pool4 = tflearn.max_pool_2d(conv4_3, 2, strides=2)
    conv5_1 = tflearn.conv_2d(pool4, 512, 3, activation='relu', name="conv5_1")
    conv5_2 = tflearn.conv_2d(conv5_1, 512, 3, activation='relu', name="conv5_2")
    conv5_3 = tflearn.conv_2d(conv5_2, 512, 3, activation='relu', name="conv5_3")
    pool5 = tflearn.max_pool_2d(conv5_3, 2, strides=2)

    fc6 = tflearn.fully_connected(pool5, 4096, activation='relu', name="fc6")
    fc6_dropout = tflearn.dropout(fc6, 0.5)

    fc7 = tflearn.fully_connected(fc6_dropout, 4096, activation='relu', name="fc7")
    fc7_droptout = tflearn.dropout(fc7, 0.5)

    fc8 = tflearn.fully_connected(fc7_droptout, 3, activation='softmax', name="fc8")

    mm = tflearn.Momentum(learning_rate=0.01, momentum=0.9, lr_decay=0.1, decay_step=1000)

    network = tflearn.regression(fc8, optimizer=mm, loss='categorical_crossentropy', restore=False)
    # Training
    model = tflearn.DNN(network)
    model.load("model_resnet_+-------cifar10-14000")
    img = load_img_label('/home/bai/最新数据/验证数据/经过预处理的原始图像')
    livermask = load_mask('/home/bai/最新数据/验证数据/mask')
    abc=[]
    np.array(abc)
    for i in range(img.shape[0]):
        a = np.zeros((512,512,3,3),dtype=np.float32)#最后一项表示第几个分割方式，倒数第二项表示预测的三个结果
        for j in range(0,3):
           this_mask =  livermask[i, :, :, j]
           this_mask_num = int(np.max(this_mask))
           for k in range(1,this_mask_num+1):
               if os.path.exists('/home/bai/最新数据/验证数据/超像素块/'+str(i)+'_'+str(j)+'_'+str(k)+'.jpg'):
                   this_img = io.imread('/home/bai/最新数据/验证数据/超像素块/'+str(i)+'_'+str(j)+'_'+str(k)+'.jpg')
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
        misc.imsave('/home/bai/PycharmProjects/DeepRESIDUALNETWORKS/result/' + str(i) + '.jpg', final)
if __name__ == '__main__':
    return2img()