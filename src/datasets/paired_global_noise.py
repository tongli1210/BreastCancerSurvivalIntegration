import torch
import math
import sys
import random
from PIL import Image
import numpy as np
import numbers
import types
import collections
import warnings

def add_noise(img, v, inplace=False):
    if not isinstance(img, torch.Tensor):
        raise TypeError('img should be Tensor Image. Got {}'.format(type(img)))
    if not inplace:
        img = img.clone()

    img = img - v
    return img

class PairedGlobalNoise(object):
    """ Randomly selects a rectangle region in an image and erases its pixels.
        'Random Erasing Data Augmentation' by Zhong et al.
        See https://arxiv.org/pdf/1708.04896.pdf
    Args:
         p: probability that the random erasing operation will be performed.
         scale: range of proportion of erased area against input image.
         ratio: range of aspect ratio of erased area.
         value: erasing value. Default is 0. If a single int, it is used to
            erase all pixels. If a tuple of length 3, it is used to erase
            R, G, B channels respectively.
            If a str of 'random', erasing each pixel with random values.
         inplace: boolean to make this transform inplace. Default set to False.

    Returns:
        Erased Image.
    # Examples:
        >>> transform = transforms.Compose([
        >>> transforms.RandomHorizontalFlip(),
        >>> transforms.ToTensor(),
        >>> transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        >>> transforms.RandomErasing(),
        >>> ])
    """

    def __init__(self, p=0.5, loc=[0.15, 0.35], scale=[0.05, 0.15], inplace=False):
        if (loc[0] > loc[1]):
            warnings.warn("range should be of kind (min, max)")
        if (scale[0] > scale[1]):
            warnings.warn("range should be of kind (min, max)")
        if p < 0 or p > 1:
            raise ValueError("range of noising probability should be between 0 and 1")

        self.p = p
        self.loc = loc
        self.scale = scale 
        self.inplace = inplace

    @staticmethod
    def get_params(img, loc, scale):        
        mu1 = random.uniform(loc[0], loc[1])
        sigma1 = random.uniform(scale[0], scale[1])
        v1 = np.random.normal(loc=mu1, scale=sigma1, size=img.shape)
        v1 = torch.from_numpy(v1).float()
        #v2 = -v1
        mu2 = random.uniform(loc[0], loc[1])
        sigma2 = random.uniform(scale[0], scale[1])
        v2 = np.random.normal(loc=mu2, scale=sigma2, size=img.shape)
        v2 = torch.from_numpy(v2).float()
        return v1, v2

    def __call__(self, img1, img2):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W) to be erased.

        Returns:
            img (Tensor): Erased Tensor image.
        """
        # force the noise and not using self.p        
        v1, v2 = self.get_params(img1, loc=self.loc, scale=self.scale)
        img1 = add_noise(img1, v1, self.inplace)
        img2 = add_noise(img2, v2, self.inplace)
        return img1, img2

