# -*- coding:utf8 -*-
import os
from skimage import io
import numpy as np
import natsort
import time
from PIL import Image
import scipy.io as sio
import glob


def load_mask_deprecated(directory, filepattern="*"):
    import numpy as np
    if not os.path.exists(directory) or not os.path.isdir(directory):
        raise ValueError("Given directory does not exist or is a file : " + str(directory))
    print('\tRead data', directory)
    lstFilesDCM = natsort.natsorted(glob.glob(os.path.join(directory, filepattern)))
    print ('\tLength dicom series', len(lstFilesDCM))
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


def load_mask(directory, filepattern="*"):
    if not os.path.isdir(directory):
        raise ValueError("Given directory does not exist or is a file : " + str(directory))
    print('\tRead data', directory)
    lstFilesDCM = natsort.natsorted(glob.glob(os.path.join(directory, filepattern)))
    print ('\tLength dicom series', len(lstFilesDCM))

    # Get ref file
    # Load dimensions based on the number of rows, columns, and slices (along the Z axis)
    # The array is sized based on 'ConstPixelDims'

    """
    ArrayDicom = np.empty(shape=[len(lstFilesDCM), 512, 512, 3], dtype=np.float16)
    # loop through all the DICOM files
    for i, filenameDCM in enumerate(lstFilesDCM):
        # read the file
        masknum = i % 3
        ds = sio.loadmat(filenameDCM)
        ArrayDicom[i / 3, :, :, masknum] = ds['label']  # 'mask'+str(masknum+1)
    return ArrayDicom
    """

    L = []
    for i in range(0, len(lstFilesDCM), 3):
        _temp_arr = np.stack([
            sio.loadmat(lstFilesDCM[i])['label'],
            sio.loadmat(lstFilesDCM[i + 1])['label'],
            sio.loadmat(lstFilesDCM[i + 2])['label'],
        ], axis=2)
        L.append(_temp_arr)
    return np.stack(L, axis=0)



def main():
    livermask = load_mask('D:/LiangData_Afterchoose/mask')
    pass


if __name__ == '__main__':
    start_time = time.time()
    main()
    end_time = time.time()
    print("time elapsed: {}s.".format(end_time - start_time))
