import PIL.Image
import torch.utils.data
import torchvision
from torch import nn
from torch.utils.tensorboard import SummaryWriter
import addnoise
import Unet_model
import dataset
import feng_model
import feng_add_kernel
import eval
import loss
import time
import path
import add_res

writer = SummaryWriter(path.log_path)
begin = time.time()
# -----------MODEL---------------
if (path.load_path == ""):
    #net = feng_model.feng_s_model()
    # net = feng_prelu.feng_s_model()
    # net = feng_model.feng_s_model()
    # net = feng_add_kernel.feng_s_model()
    net = add_res.model()
    # net = Unet_model.model()
else:
    net = torch.load(path.load_path)
net = net.cuda()
# print(net)
# ---------LOSS FUNCTION-------------------------------
# loss_func = nn.MSELoss()
loss_func = nn.L1Loss()
# loss_func = loss.myloss()
loss_func = loss_func.cuda()
# -----------hyper params---------------
lr = 1e-3
batch_size = 128
epoch = 60
trained = 0
optimizer = torch.optim.Adam(net.parameters(), lr)

# -------------data pre-process------------
'''train_set = dataset.mydataset(path.dir,100,False)
print("read ok , spent time:{:.2f}s".format(time.time()-begin))
train_loader = torch.utils.data.DataLoader(train_set,batch_size=batch_size,shuffle=True,)'''

'''print("preparation cost:{:.2f}s".format(time.time()-begin))
last = time.time()
train_set = dataset.mydataset(path.dir,crop_all=True)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
print("read ok , spent time:{:.2f}s".format(time.time() - last))
'''
for _ in range(epoch):
    if _ < trained:
        continue
    # reload dataset every epoch , crop 50*50 randomly every time
    last = time.time()
    train_set = dataset.mydataset(path.dir)
    print("read ok , spent time:{:.2f}s".format(time.time() - last))
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, )
    last = time.time()
    net = net.train()
    tt_loss = 0
    for bch_img in train_loader:
        # 克隆数据
        imgs = bch_img.clone().cuda()
        for img in imgs:
            img = addnoise.add_noise_cuda(img)
        # 原图
        bch_img = bch_img.cuda()
        # 降噪
        out = net(imgs)
        # 计算，更新
        loss = loss_func(out * 255, bch_img * 255)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        tt_loss += loss
    tt_loss /= len(train_set)
    # save the model
    # if (_+1)%10 ==0:
    torch.save(net, path.save_path)
    writer.add_scalar("log", tt_loss, _)
    print("epoch {} , loss:{} , spend time : {:.2f}second\n".format(_ + 1, tt_loss, time.time() - last))
    if (_ + 1) == 30:
        lr = 1e-4
        optimizer = torch.optim.Adam(net.parameters(), lr)
    if (_ + 1) == 60:
        lr = 1e-5
        optimizer = torch.optim.Adam(net.parameters(), lr)
writer.close()
print("total train time:{:.2f}s".format(time.time() - begin))
