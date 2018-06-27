#!/usr/bin/python
# -*- coding:utf8 -*-
import numpy as np
from skimage import io
from scipy import misc
import scipy.ndimage as sp
import scipy.misc as scm
import scipy.io as sio
import os
import natsort
import glob
from matplotlib import pyplot as plt
from Pretreatment import read_dicom_series, step1_preprocess_img_slice


def loaddata():
    plt.set_cmap('gray')

    # 读取训练肝脏标准
    collmask = []
    collkk = []
    for i in range(1, 2):
        stk = '/home/bai/3Dircadb1/3Dircadb1.' + str(i) + '/MASKS_DICOM/liver'
        print(stk)
        collk = read_dicom_series(stk)
        for j in range(collk.shape[2]):
            # collkk[...,j] = step1_preprocess_img_slice(collk[...,j])
            collkk.append(step1_preprocess_img_slice(collk[..., j]))
            # collk[...,j] = np.resize(collk[...,j],(224,224))
        collmask.append(collkk)
    collmask = np.concatenate(collmask[:], axis=2)
    # collmask=np.transpose(collmask,(2,0,1))
    collmask = np.reshape(collmask, (collmask.shape[0], collmask.shape[1], collmask.shape[2], 1))
    collmask = np.array(collmask, dtype='float64')

    # 读取训练肝脏图像
    coll = []
    collkk = []
    for i in range(1, 2):
        stk = '/home/bai/3Dircadb1/3Dircadb1.' + str(i) + '/PATIENT_DICOM/'
        print(stk)
        collk = read_dicom_series(stk)
        for j in range(collk.shape[2]):
            collkk.append(step1_preprocess_img_slice(collk[..., j]))
        #     io.imsave('/home/bai/alias/'+np.str(i)+'.tif',img_p)
        # misc.imsave('/home/bai/preprogress/'+str(i+1)+'_'+str(j+1)+'.tif',img_p)
        coll.append(collkk)
    coll = np.concatenate(coll[:], axis=2)
    # coll=np.transpose(coll,(2,0,1))
    coll = np.reshape(coll, (coll.shape[0], coll.shape[1], coll.shape[2], 1))
    coll = np.array(coll, dtype='float64')

    # 读取测试肝脏图像
    testcoll = []
    collkk = []
    stk = '/home/bai/3Dircadb1.20/PATIENT_DICOM'
    print(stk)
    collk = read_dicom_series(stk)
    for j in range(collk.shape[2]):
        collkk.append(step1_preprocess_img_slice(collk[..., j]))
        # collk[..., j] = np.resize(collk[..., j], (224, 224))
    testcoll.append(collkk)
    testcoll = np.concatenate(testcoll[:], axis=2)
    # testcoll=np.transpose(testcoll,(2,0,1))
    testcoll = np.reshape(testcoll, (testcoll.shape[0], testcoll.shape[1], testcoll.shape[2], 1))
    testcoll = np.array(testcoll, dtype='float64')

    # 读取测试肝脏标准
    testcollmask = []
    collkk = []
    stk = '/home/bai/3Dircadb1.20/MASKS_DICOM/liver'
    print(stk)
    collk = read_dicom_series(stk)
    for j in range(collk.shape[2]):
        collkk.append(step1_preprocess_img_slice(collk[..., j]))
        # collk[..., j] = np.resize(collk[..., j], (224, 224))
    testcollmask.append(collkk)
    testcollmask = np.concatenate(testcollmask[:], axis=2)
    # testcollmask=np.transpose(testcollmask,(2,0,1))
    testcollmask = np.reshape(testcollmask, (testcollmask.shape[0], testcollmask.shape[1], testcollmask.shape[2], 1))
    testcollmask = np.array(testcollmask, dtype='float64')

    # 预留读取训练肝脏肿瘤标准

    # 预留读取测试肝脏肿瘤标准
    return coll, collmask, testcoll, testcollmask


# 将超像素对回的原图像进行裁剪，填充
def imshow(*args, **kwargs):
    """ Handy function to show multiple plots in on row, possibly with different cmaps and titles
    Usage:
    imshow(img1, title="myPlot")
    imshow(img1,img2, title=['title1','title2'])
    imshow(img1,img2, cmap='hot')
    imshow(img1,img2,cmap=['gray','Blues']) """
    cmap = kwargs.get('cmap', 'gray')
    title = kwargs.get('title', '')
    if len(args) == 0:
        raise ValueError("No images given to imshow")
    elif len(args) == 1:
        plt.title(title)
        plt.imshow(args[0], interpolation='none')
    else:
        n = len(args)
        if type(cmap) == str:
            cmap = [cmap] * n
        if type(title) == str:
            title = [title] * n
        plt.figure(figsize=(n * 5, 10))
        for i in range(n):
            plt.subplot(1, n, i + 1)
            plt.title(title[i])
            plt.imshow(args[i], cmap[i])
    plt.show()


# img是已经将mask对回原图的结果
# 还要改！！！！
def Cropped_fill(img, mask, i):
    # 找到图像非零元素边界
    x = np.where(mask == i)[0]
    y = np.where(mask == i)[1]
    # [x,y] = img.find(img!=0)
    xmin = min(x)
    xmax = max(x)
    ymin = min(y)
    ymax = max(y)
    width = max(xmax - xmin, ymax - ymin)
    length = width / 2
    middlex = (xmin + xmax) / 2
    middley = (ymin + ymax) / 2
    if middlex - length < 0 or  middley - length < 0:
        Nimg = img[xmin:xmin + width, ymin:ymin + width]
    elif middlex + length > 512 or middley + length > 512:
        Nimg = img[xmax-width:xmax, ymax-width:ymax]
    else:
        Nimg = img[middlex - length:middlex + length, middley - length:middley + length]
    '''
    #矩阵中非零点的个数
    num_nonzero = len(x)
    #矩阵中所有非零点的和
    sum_nonzero = np.sum(Nimg)
    #填充的值
    average = sum_nonzero/num_nonzero
    Nimg[Nimg==0] = average
    '''
    # Nimg = Nimg.resize((32,32))#这里可以指定矩阵类型
    Nimg = scm.imresize(Nimg, (32, 32), 'bicubic')
    return Nimg


# 读取所有图像
def load_img_label(directory, filepattern="*"):
    if not os.path.isdir(directory):
        raise ValueError("Given directory does not exist or is a file : {}".format(directory))

    print('\tRead data', directory)
    lstFilesDCM = [os.path.join(directory, _) for _ in os.listdir(directory)]
    lstFilesDCM.sort(key=lambda _: int(os.path.split(_.split(".")[0])[-1]))
    # lstFilesDCM = natsort.natsorted(glob.glob(os.path.join(directory, filepattern)))
    print('\tLength dicom series', len(lstFilesDCM))

    # Get ref file
    # Load dimensions based on the number of rows, columns, and slices (along the Z axis)
    # The array is sized based on 'ConstPixelDims'

    # ConstPixelDims = (len(lstFilesDCM), 512, 512, 1)  # 下次训练时要改成512×512
    # ArrayDicom = np.zeros(ConstPixelDims, dtype=RefDs.pixel_array.dtype)

    """
    ArrayDicom = np.empty(shape=[len(lstFilesDCM), 512, 512, 1], dtype=np.float16)
    # loop through all the DICOM files
    for filenameDCM in lstFilesDCM:
        # read the file
        ds = io.imread(filenameDCM)
        # store the raw image data
        ArrayDicom[lstFilesDCM.index(filenameDCM), :, :, 0] = ds
    return ArrayDicom
    """

    return np.stack([np.expand_dims(io.imread(_), axis=2) for _ in lstFilesDCM], axis=0)


def load_mask(directory, filepattern="*"):
    if not os.path.exists(directory) or not os.path.isdir(directory):
        raise ValueError("Given directory does not exist or is a file : " + str(directory))
    print('\tRead data', directory)
    lstFilesDCM = natsort.natsorted(glob.glob(os.path.join(directory, filepattern)))
    print('\tLength dicom series', len(lstFilesDCM))

    """
    # Get ref file
    # Load dimensions based on the number of rows, columns, and slices (along the Z axis)
    ConstPixelDims = (len(lstFilesDCM), 512, 512, 3)
    # The array is sized based on 'ConstPixelDims'
    ArrayDicom = np.zeros(ConstPixelDims, dtype=np.float16)
    masknum = 0
    # loop through all the DICOM files
    for filenameDCM in lstFilesDCM:
        # read the file
        ds = sio.loadmat(filenameDCM)
        ArrayDicom[lstFilesDCM.index(filenameDCM) / 3, :, :, masknum] = ds['label']  # 'mask'+str(masknum+1)
        masknum = masknum + 1
        if masknum == 3:
            masknum = 0
        # store the raw image data

    return ArrayDicom
    """
    # 如果没有缓存，创建缓存
    cache_dir = os.path.join(os.path.split(directory)[0], 'mask_cache')
    if not os.path.isdir(cache_dir):
        os.mkdir(cache_dir)
        # 创建缓存
        print("creating cache: {}".format(cache_dir))
        L = []
        for i in range(0, len(lstFilesDCM), 3):
            _temp_arr = np.stack([
                sio.loadmat(lstFilesDCM[i])['label'],
                sio.loadmat(lstFilesDCM[i + 1])['label'],
                sio.loadmat(lstFilesDCM[i + 2])['label'],
            ], axis=2)
            L.append(_temp_arr)
        np.save(os.path.join(cache_dir, "mask_arr.npy"), np.stack(L, axis=0))
    # 读缓存
    return np.load(os.path.join(cache_dir, "mask_arr.npy"))


# 将mask对到原图、肝脏标签和肿瘤标签上
# def mask2original(img,mask):
def return2img():
    img = load_img_label('D:/LiangData_Afterchoose/img')
    liverlabe = load_img_label('D:/LiangData_Afterchoose/liver')
    livertumorlabel = load_img_label('D:/LiangData_Afterchoose/lesion')
    livermask = load_mask('D:/LiangData_Afterchoose/mask')
    # livermask = load_mask('/home/bai/cxs/mask2')
    finalmask = []
    for i in range(img.shape[0]):
        if len(finalmask) < 450000:
            my_img = img[i, :, :, 0]
            this_liver = liverlabe[i, :, :, 0]
            this_tumor = livertumorlabel[i, :, :, 0]
            for j in range(0, 3):
                this_mask = livermask[i, :, :, j]
                this_mask_num = int(np.max(this_mask))
                for k in range(1, this_mask_num + 1):
                    # this_img[np.where(this_mask == k)] = my_img[np.where(this_mask == k)] # todo: remove
                    # this_img = my_img * (this_mask == k)
                    #img2save = Cropped_fill(this_img, this_mask, k)
                    img2save = Cropped_fill(my_img, this_mask, k)
                    if len(np.nonzero(img2save)[0]) != 0:
                        # liver probability
                        livertotalpixel = np.sum(this_mask == k)
                        liverfindpixel = np.sum(this_liver * (this_mask == k) > 0)
                        liver_probability = 0 if liverfindpixel == 0 or livertotalpixel == 0 \
                            else float(liverfindpixel) / livertotalpixel
                        # tumor probability
                        tumortotalpixel = np.sum(this_mask == k)
                        tumorfindpixel = np.sum(this_tumor * (this_mask == k) > 0)
                        tumor_probability = 0 if tumorfindpixel == 0 or tumortotalpixel == 0 \
                            else float(tumorfindpixel) / tumortotalpixel
                        # save images
                        if tumor_probability > 0.3:
                            finalmask.append((0, 0, 1))
                            misc.imsave('D:/LiangData_Afterchoose/superpixel/{}_{}_{}.jpg'.format(i, j, k), img2save)
                        elif liver_probability > 0.5:
                            if not finalmask:
                                finalmask = [(0, 1, 0)]
                            if np.sum(np.array(finalmask) * (0, 1, 0)) < 150000:
                                finalmask.append((0, 1, 0))
                                misc.imsave('D:/LiangData_Afterchoose/superpixel/{}_{}_{}.jpg'.format(i, j, k),
                                            img2save)
                        else:
                            if not finalmask:
                                finalmask = [(0, 1, 0)]
                            if np.sum(np.array(finalmask) * (1, 0, 0)) < 150000:
                                finalmask.append((1, 0, 0))
                                misc.imsave('D:/LiangData_Afterchoose/superpixel/{}_{}_{}.jpg'.format(i, j, k),
                                            img2save)
            print("第 %d 张图像完成" % (i))
    # np.save('/home/bai/cxs/cxs/mask/finalmask',finalmask)
    np.save('D:/LiangData_Afterchoose/superpixellabel/finalmask', finalmask)


if __name__ == '__main__':
    return2img()
# ruarua = Cropped_fill(rua[...,0])
# imshow(ruarua)
# from scipy import misc
# #用来保存图像
# misc.imsave('/home/bai/preprogress/rua/1.jpg',ruarua)
