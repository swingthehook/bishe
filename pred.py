import time
import torch
import torchvision
import addnoise
import loss
import path
import PIL
import eval


net_path = "ad_L1.pth"
net = torch.load(net_path)
net = net.cpu()
net = net.eval()
# pred = time.time()
# read lena
org = PIL.Image.open("LenaRGB.bmp")
#    add the noise0
org = torchvision.transforms.ToTensor()(org)
pred = time.time()
noised = addnoise.add_noise(org,15)
tmp = torchvision.transforms.ToPILImage()(noised)
tmp.save("noised.jpg")
#   go through the model
res = noised.unsqueeze(0)
res = net(res)
res[res > 1] = 1
res[res < 0] = 0
res = res.squeeze(0)
eval.eval(org, res)
torchvision.transforms.ToPILImage()(res).save("result.jpg")

print("prediction spend {:.2f} s".format(time.time() - pred))
