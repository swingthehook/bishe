import torch
import random
import numpy as np


def add_noise(img, lamb=None):
    img = img.clone()
    if lamb == None:
        level = [15, 25, 50]
        n = random.choice(level)
    else:
        n = lamb
    noise = torch.Tensor(np.random.normal(0, n, img.shape))
    noise /= 255
    img += noise
    img[img > 1] = 1
    img[img < 0] = 0
    return img


def add_noise_cuda(img, lamb=None):
    if lamb == None:
        level = [15, 25, 50]
        n = random.choice(level)
    else:
        n = lamb
    noise = torch.Tensor(np.random.normal(0, n, img.shape)).cuda()
    noise /= 255
    img += noise
    img[img > 1] = 1
    img[img < 0] = 0


def leveled_add_noise(img, level):
    noise = torch.randn(size=img.shape, dtype=torch.float32)
    noise *= level
    img += noise
    img[img > 1] = 1
    img[img < 0] = 0
    return img
