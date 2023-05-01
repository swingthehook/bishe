import torch
from torch import nn
import torch.nn.functional as F
import augmentation


class mul(nn.Module):
    def __init__(self):
        super().__init__()
        self.p1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=24, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(24),
            nn.ReLU()
        )
        self.p2 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=20, kernel_size=5, padding=2, stride=1),
            nn.BatchNorm2d(20),
            nn.ReLU()
        )
        self.p3 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=20, kernel_size=7, padding=3, stride=1),
            nn.BatchNorm2d(20),
            nn.ReLU()
        )

    def forward(self, img):
        res1 = self.p1(img)
        res2 = self.p2(img)
        res3 = self.p3(img)
        return torch.cat((res1, res2, res3), dim=1)


class sub(nn.Module):
    def __init__(self, para):
        super().__init__()
        self.net = nn.Sequential()
        for i in range(14):
            self.net.append(nn.Sequential(
                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, dilation=1),
                nn.BatchNorm2d(64),
                nn.ReLU()
            ))
        for pair in para:
            self.net[pair[0] - 1][0] = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1,
                                                 padding=pair[1], dilation=pair[1])
        self.net.append(nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=3, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(3),
            nn.ReLU()
        ))

    def forward(self, img):
        return self.net(img)


class PAM(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, img):
        b, c, h, w = img.shape
        f1 = f2 = f3 = img.reshape(b, c, h * w)
        temp = f1.permute(0, 2, 1).matmul(f2)
        temp = torch.exp(temp)
        for i in temp:
            i /= i.sum()
        temp = f3.matmul(temp).reshape(b, c, h, w)
        return temp


class CAM(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, img):
        b, c, h, w = img.shape
        f1 = f2 = f3 = img.reshape(b, c, h * w)
        temp = f1.matmul(f2.permute(0, 2, 1))
        temp = torch.exp(temp)
        for i in temp:
            i /= i.sum()
        temp = temp.matmul(f3).reshape(b, c, h, w)
        return temp



class attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=9, out_channels=3, kernel_size=3, stride=1, padding=1)
        self.tanh = nn.Tanh()
        self.conv2 = nn.Conv2d(in_channels=9, out_channels=3, kernel_size=3, stride=1, padding=1)
        self.ksx = nn.Parameter(torch.Tensor([
            [1, 0, -1],
            [2, 0, -2],
            [1, 0, -1]
        ]).unsqueeze(dim=0).unsqueeze(dim=0).expand(1, 3, 3, 3), requires_grad=False
                                )
        self.ksy = nn.Parameter(
            torch.Tensor([
                [1, 2, 1],
                [0, 0, 0],
                [-1, -2, -1]
            ]).unsqueeze(dim=0).unsqueeze(dim=0).expand(1, 3, 3, 3), requires_grad=False
        )
        self.klap = nn.Parameter(
            torch.Tensor([
                [0, 1, 0],
                [1, -4, 1],
                [0, 1, 0]
            ]).unsqueeze(dim=0).unsqueeze(dim=0).expand(1, 3, 3, 3), requires_grad=False
        )

    def forward(self, temp, img):
        temp1 = self.conv1(temp)
        m1 = F.conv2d(img, self.ksx, stride=1, padding=1)
        m2 = F.conv2d(img, self.ksy, stride=1, padding=1)
        m3 = F.conv2d(img, self.klap, stride=1, padding=1)
        temp2 = torch.cat((temp1, img, m1, m2, m3), dim=1)
        temp2 = self.tanh(temp2)
        temp2 = self.conv2(temp2)
        temp1 = temp1 * temp2
        return temp1


class model(nn.Module):
    def __init__(self):
        super().__init__()
        self.mul1 = mul()
        self.mul2 = mul()
        self.mul3 = mul()
        self.sub1 = sub([[1, 2], [6, 2], [3, 3], [8, 3]])
        self.sub2 = sub([[2, 3], [7, 3], [5, 2], [10, 2]])
        self.sub3 = sub([[3, 2], [6, 3], [9, 2], [12, 3]])
        self.attention = attention()

    def forward(self, img):
        up, img, down = augmentation.img_aug(img)
        prl1 = self.sub1(self.mul1(up))
        prl2 = self.sub2(self.mul2(img))
        prl3 = self.sub3(self.mul3(down))
        prl1, prl2, prl3 = augmentation.restore(prl1, prl2, prl3)
        temp = torch.cat((prl1, prl2, prl3), dim=1)
        temp = self.attention(temp, img)
        img = img - temp
        return img


'''img = torch.ones((2, 1, 3, 3))
print(PAM()(img))
print(CAM()(img))'''
