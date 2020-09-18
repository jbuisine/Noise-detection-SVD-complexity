# main imports
import numpy as np
import sys

# image transform imports
from PIL import Image
from skimage import color

from ipfml.processing import transform, compression, segmentation
from ipfml.filters import convolution, kernels
from ipfml import utils

def get_features(data_type, block):
    """
    Method which returns the data type expected
    """

    if data_type == 'svd_entropy':

        l_img = transform.get_LAB_L(block)

        blocks = segmentation.divide_in_blocks(l_img, (20, 20))

        values = []
        for b in blocks:
            sv = compression.get_SVD_s(b)
            values.append(utils.get_entropy(sv))
        data = np.array(values)

    if data_type == 'svd_entropy_noise':

        l_img = transform.get_LAB_L(block)

        blocks = segmentation.divide_in_blocks(l_img, (20, 20))

        values = []
        for b in blocks:
            sv = compression.get_SVD_s(b)
            sv_size = len(sv)
            values.append(utils.get_entropy(sv[int(sv_size / 4):]))
        data = np.array(values)
        
    return data


