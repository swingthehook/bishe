import torch
from torch import nn


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


class attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=6, out_channels=3, kernel_size=3, stride=1, padding=1)
        self.tanh = nn.Tanh()
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=3, kernel_size=3, stride=1, padding=1)

    def forward(self, temp, img):
        temp1 = self.conv1(temp)
        temp2 = torch.cat((temp1, img), dim=1)
        temp2 = self.tanh(temp2)
        temp2 = self.conv2(temp2)
        temp1 = temp1 * temp2
        return temp1


class feng_s_model(nn.Module):
    def __init__(self):
        super().__init__()
        self.mul1 = mul()
        self.mul2 = mul()
        self.sub1 = sub([[1, 2], [6, 2], [3, 3], [8, 3]])
        self.sub2 = sub([[2, 3], [7, 3], [5, 2], [10, 2]])
        self.attention = attention()

    def forward(self, img):
        prl1 = self.sub1(self.mul1(img))
        prl2 = self.sub2(self.mul2(img))
        temp = torch.cat((prl1, prl2), dim=1)
        temp = self.attention(temp, img)
        img = img - temp
        return img


'''model = feng_s_model()
test = torch.rand(2,3,50,50)
test = model(test)'''
