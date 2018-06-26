#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import
from loadData import loaddata
"""
Created on Sat Jul  2 14:58:30 2016

@author: ubuntu
"""


""" Very Deep Convolutional Networks for Large-Scale Visual Recognition.
Applying VGG 16-layers convolutional network to Oxford‘s 17 Category Flower
Dataset classification task.
References:
    Very Deep Convolutional Networks for Large-Scale Image Recognition.
    K. Simonyan, A. Zisserman. arXiv technical report, 2014.
Links:
    http://arxiv.org/pdf/1409.1556
"""

# 在Ubuntu的terminal中运行是偶尔会报错关于PIL的，但是如此使用就不会报错了
from PIL import Image

#a = Image.open('')

import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
import numpy as np


def load_image(in_image):
    """ Load an image, returns PIL.Image. """
    img = Image.open(in_image)
    return img


#img_path =''
#img = load_image(img_path)


def resize_image(in_image, new_width, new_height, out_image=None,
                 resize_mode=Image.ANTIALIAS):
    """ Resize an image.

    Arguments:
        in_image: `PIL.Image`. The image to resize.
        new_width: `int`. The image new width.
        new_height: `int`. The image new height.
        out_image: `str`. If specified, save the image to the given path.
        resize_mode: `PIL.Image.mode`. The resizing mode.

    Returns:
        `PIL.Image`. The resize image.

    """
    img = in_image.resize((new_width, new_height), resize_mode)
    if out_image:
        img.save(out_image)
    return img


#img = resize_image(img, 224, 224)


def pil_to_nparray(pil_image):
    """ Convert a PIL.Image to numpy array. """
    pil_image.load()
    return np.asarray(pil_image, dtype="float32")


#img = pil_to_nparray(img)
print(u'用于测试的图片加载完成！')
# Data loading and preprocessing
import tflearn.datasets.oxflower17 as oxflower17
(coll,collmask,testcoll,testcollmask) = loaddata()

#print('------')
#print('666666666666')
#X, Y = oxflower17.load_data(one_hot=True)

# Building ‘VGG Network‘以下为模型的加载，其中3是卷积核的大小即3*3.64/128/256/512是卷积核的个数
network = input_data(shape=[None, 224, 224, 1])

network = conv_2d(network, 64, 3, activation='relu')
network = conv_2d(network, 64, 3, activation='relu')
network = max_pool_2d(network, 2, strides=2)

network = conv_2d(network, 128, 3, activation='relu')
network = conv_2d(network, 128, 3, activation='relu')
network = max_pool_2d(network, 2, strides=2)

network = conv_2d(network, 256, 3, activation='relu')
network = conv_2d(network, 256, 3, activation='relu')
network = conv_2d(network, 256, 3, activation='relu')
network = max_pool_2d(network, 2, strides=2)

network = conv_2d(network, 512, 3, activation='relu')
network = conv_2d(network, 512, 3, activation='relu')
network = conv_2d(network, 512, 3, activation='relu')
network = max_pool_2d(network, 2, strides=2)

network = conv_2d(network, 512, 3, activation='relu')
network = conv_2d(network, 512, 3, activation='relu')
network = conv_2d(network, 512, 3, activation='relu')
network = max_pool_2d(network, 2, strides=2)

network = fully_connected(network, 4096, activation='relu')
network = dropout(network, 0.5)
network = fully_connected(network, 4096, activation='relu')
network = dropout(network, 0.5)
network = fully_connected(network, 17, activation='softmax')

network = regression(network, optimizer='rmsprop',
loss ='categorical_crossentropy',
learning_rate = 0.001)

# Training
# max_checkpoints是存储checkpoint文件的个数，如果超过个数，应该是自动删除
model = tflearn.DNN(network, checkpoint_path='model_vgg',
max_checkpoints = 1, tensorboard_verbose = 0)
# snapshot_step表示执行多少步后保存checkpoint文件,n_epoch是执行循环的次数，batch_size每次读取图片的个数,如果内存不足可以通过这个进行调节。
print(u'开始加载模型')
# model.load(‘/home/ubuntu/pythonproject/tflearnproject/model_vgg-20‘)
# model.load(‘model_vgg-30‘)
model.fit(coll, collmask, n_epoch=1, shuffle=True,
          show_metric=True, batch_size=8, snapshot_step=10,
          snapshot_epoch=False, run_id='vgg_oxflowers17')
model.save('vgg16.tflearn')
# model.predit(X[0])
print(u'开始预测')
model.predict(testcoll)
# model.load('vgg16.tflearn')