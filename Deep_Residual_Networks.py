#!/usr/bin/python
# -*- coding:utf8 -*-
from __future__ import division, print_function, absolute_import
from loadData import load_img_label
import numpy as np
import tensorflow as tf
import os
import natsort
import glob
from skimage import io
from keras.layers import merge
from keras.layers.convolutional import Convolution2D, MaxPooling2D,ZeroPadding2D,AveragePooling2D
from keras.layers.core import Dense, Activation,Flatten
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.layers import Input

#读入经过预处理的数据
#(coll,collmask,testcoll,testcollmask) = loaddata()



    #testcollmask = np.load('/home/bai/PycharmProjects/DeepRESIDUALNETWORKS/最新数据/训练数据/超像素标签/finalmask.npy')
# def identity_block(x, nb_filter, kernel_size=3):
#     k1, k2, k3 = nb_filter
#     out = Convolution2D(k1, 1, 1)(x)
#     out = BatchNormalization()(out)
#     out = Activation('relu')(out)
#
#     out = Convolution2D(k2, kernel_size, kernel_size, border_mode='same')(out)
#     out = BatchNormalization()(out)
#     out = Activation('relu')(out)
#
#     out = Convolution2D(k3, 1, 1)(out)
#     out = BatchNormalization()(out)
#
#     out = merge([out, x], mode='sum')
#     out = Activation('relu')(out)
#     return out


import tflearn
import dicom
# Residual blocks
# 32 layers: n=5, 56 layers: n=9, 110 layers: n=18
n = 5

# Data loading
from tflearn.datasets import cifar10

# coll=np.array(coll,dtype='float64')
# X = coll

# Y = collmask
# #Y=np.array(Y,dtype='float64')
# testX = testcoll
# #testX=np.array(testX,dtype='float64')
# testY = testcollmask
# #testY=np.array(testY,dtype='float64')
# X = k
# Y = k
# testX =k
# testY =k
# (X, Y), (testX, testY) = cifar10.load_data()
#collmask = tflearn.data_utils.to_categorical(collmask, 3)
#testcollmask = tflearn.data_utils.to_categorical(testcollmask, 3)
def load_img(path):
    ConstPixelDims = (len(path), 32, 32, 1)  # 下次训练时要改成512×512
    ArrayDicom = np.zeros(ConstPixelDims, dtype=np.float16)
    for filenameDCM in path:
        # read the file
        ds = io.imread(filenameDCM)
        # store the raw image data
        ArrayDicom[path.index(filenameDCM), :, :, 0] = ds
        # store the raw image data
        return ArrayDicom
# Real-time data preprocessing
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
model = tflearn.DNN(net, checkpoint_path='model_resnet_+-------cifar10',
                    max_checkpoints=10, tensorboard_verbose=3,
                    clip_gradients=0.)#这里改成啦3
#session=tf.Session("grpc//172.16.10.50:2222")
#validation_set=(testcoll, testcollmask),
print('1')
collmask = np.load('/media/bai/Elements/LiangData_Afterchoose/superpixellabel/finalmask.npy')
print('2')
lstFilesDCM = natsort.natsorted(glob.glob(os.path.join('/media/bai/Elements/LiangData_Afterchoose/superpixel', '*')))
for i in range(0,100):
    #这里加一个生成随机数
    #coll = load_img_label('/media/bai/Elements/LiangData_Afterchoose/superpixel')
    coll = load_img(lstFilesDCM[i*3700:i*3700+3700])
    thismask = collmask[i*3700:i*3700+3700,:]
    #testcoll = load_img_label('/home/bai/cxs/cxs/testimg')
    model.fit(coll, thismask, n_epoch=5,
              snapshot_epoch=False, snapshot_step=500,
              show_metric=True, batch_size=300, shuffle=True,
              run_id=None)
model.save('my_ResNet')
print('开始预测')
#result = model.predict(testcoll)
#np.save('/home/bai/PycharmProjects/DeepRESIDUALNETWORKS/result',result)


# model.save() 用来保存模型
# model.predict()