import torch
import torchvision.io
from torch.utils.data import Dataset
from   PIL import Image
import os
import time


class mydataset(Dataset):
    def __init__(self,dir,crop_size=50,crop_num = 20,crop_all = False,eval=False):
        self.imgs = []
        totensor = torchvision.transforms.ToTensor()
        crop = torchvision.transforms.RandomCrop((crop_size,crop_size))
        for file in os.listdir(dir):
            img = Image.open(dir+"/"+file)
            if crop_all:
                height, width = img.size
                i=0
                while i+50 < width:
                    j=0
                    while j+50 <height:
                        tmp = img.crop((i,j,i+50,j+50))
                        tmp = totensor(tmp)
                        self.imgs.append(tmp)
                        j+=50
                    i+=50
            else:
                if eval:
                    img = totensor(img)
                    self.imgs.append(img)
                else:
                    for i in range(crop_num):
                        tmp = crop(img)
                        tmp = totensor(tmp)
                        self.imgs.append(tmp)


    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, item):
        return self.imgs[item]


'''begin = time.time()
set = mydataset("./pristine_images")
print("spent time :{}s".format(time.time()-begin))
'''