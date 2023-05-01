import PIL
import numpy as np
import torch
import torchvision
import loss
import dataset
import addnoise
import time
import skimage


def test(net,img):
    pred = time.time()
    net = net.eval()
    tmp = torchvision.transforms.ToPILImage()(img)

    #    add the noise
    img = addnoise.add_noise(img)
    tmp = torchvision.transforms.ToPILImage()(img)
    tmp.save("noised.jpg")
    #    through the model
    img = img.unsqueeze(0)
    img = net(img)
    img = img.squeeze(0)
    tmp = torchvision.transforms.ToPILImage()(img)
    tmp.save("result.jpg")
    print("prediction spend {} s".format(time.time() - pred))

def eval(img,repaired_img):
    trans = torchvision.transforms.ToPILImage()
    img = np.array(trans(img))
    repaired_img = np.array(trans(repaired_img))
    loss1 = skimage.metrics.peak_signal_noise_ratio(img,repaired_img)
    loss2 = skimage.metrics.structural_similarity(img,repaired_img,channel_axis=2)
    print("psnr :{} and ssim:{}".format(loss1,loss2))
