import random
import PIL
import cv2
import torch
import torchvision
import numpy
from torch import nn

import Unet_model
import augmentation
import addnoise
import dataset
import path
import torch.utils.tensorboard as TB

'''writer = TB.SummaryWriter("./test_log")
for i in range(100):
    writer.add_scalar("rand2",random.randint(0,1e5),i)



writer.close()'''
print(torch.version.cuda)
print(torch.version.__version__)


