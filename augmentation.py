import PIL.Image
import cv2
import numpy
import torch
import torchvision
import torch.nn.functional as F


def img_aug(img):
    up = F.interpolate(img, scale_factor=2, mode='bicubic')
    down = F.interpolate(img, scale_factor=0.5, mode='nearest')
    return up, img, down


def restore(up, img, down):
    up = F.interpolate(up, scale_factor=0.5, mode='nearest')
    down = F.interpolate(down, scale_factor=2, mode='bicubic')
    return up, img, down
