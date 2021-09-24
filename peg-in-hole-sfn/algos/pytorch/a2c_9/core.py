import numpy as np
import scipy.signal
from gym.spaces import Box, Discrete

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical
  
 
def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)

# class mlp(nn.Module):
#     """docstring for mlp"""
#     def __init__(self, sizes, activation, output_activation=nn.Identity):
#         super(mlp, self).__init__()

#         self.rnn = nn.GRU(sizes[0], sizes[1], 1, batch_first=True)
#         self.relu = nn.ReLU()
#         layers = []
#         for j in range(1, len(sizes)-1):
#             act = activation if j < len(sizes)-2 else output_activation
#             layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
#         self.mlp_fc = nn.Sequential(*layers)
        
#     def forward(self, x):
#         _, h = self.rnn(x)
#         h = self.relu(h.squeeze(0))
#         return self.mlp_fc(h)



def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])


def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.

    input: 
        vector x, 
        [x0, 
         x1, 
         x2]

    output:
        [x0 + discount * x1 + discount^2 * x2,  
         x1 + discount * x2,
         x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)


    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)



class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits



 
class A2C(nn.Module):
    """docstring for A2C"""
    def __init__(self, action_space, pretrained_unet=None):
        super(A2C, self).__init__()
        if pretrained_unet:
            self.unet = torch.load(pretrained_unet)
            for p in self.unet.parameters():
                p.requires_grad = False
            print("load pretrained_unet and fix parameters")
        else:
            self.unet = UNet(1,3)
        self.pool2d = nn.AdaptiveAvgPool2d((84,84))
        self.action_space = action_space
        if isinstance(self.action_space, Discrete):
            self.fc_logits = nn.Linear(3*84*84, self.action_space.n)
        elif isinstance(self.action_space, Box):
            log_std = -0.5 * np.ones(self.action_space.shape[0], dtype=np.float32)
            self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))
            self.fc_mu = nn.Linear(3*84*84, self.action_space.shape[0])
        self.fc_v = nn.Linear(3*84*84, 1)


    # collecting data
    def step(self, obs_img):
        with torch.no_grad():
            logits = self.unet(obs_img)
            logits = self.pool2d(logits)
            n,c,h,w = logits.size()
            logits = logits.view(n,-1)
            feas = torch.chunk(logits, 2, 0)
            fea = feas[1]-feas[0]

            if isinstance(self.action_space, Discrete):
                logits = self.fc_logits(fea)
                pi = Categorical(logits=logits)
            elif isinstance(self.action_space, Box):
                mu = self.fc_mu(fea)
                std = torch.exp(self.log_std)
                pi = Normal(mu, std)

            a = pi.sample()
            v = self.fc_v(fea).squeeze(-1) 
        return a, v


    # training network with a2c loss
    def forward(self, obs_img, act):
        logits = self.unet(obs_img)
        logits = self.pool2d(logits)
        n,c,h,w = logits.size()
        logits = logits.view(n,-1)
        feas = torch.chunk(logits, 2, 0)
        fea = feas[1] - feas[0]

        if isinstance(self.action_space, Discrete):
            logits_ = self.fc_logits(fea)
            pi = Categorical(logits=logits_)
        elif isinstance(self.action_space, Box):
            mu = self.fc_mu(fea)
            std = torch.exp(self.log_std)
            pi = Normal(mu, std)

        logp = pi.log_prob(act)
        v = self.fc_v(fea).squeeze(-1)
        return logp, v


    def feature_extractor(self, obs_img):
        logits = self.unet(obs_img)
        logits = self.pool2d(logits)
        n,c,h,w = logits.size()
        logits = logits.view(n,-1)
        return logits


    # test policy
    def act(self, obs_img):
        return self.step(obs_img)[0]



if __name__ == '__main__':
    action_space1 = Box(low=np.float32(-1), high=np.float32(1), shape=(1,), dtype=np.float32)
    action_space2 = Discrete(4)
    model = A2C(action_space2)
    img = torch.ones([1,1,200,250])
    log_p, v = model(img, torch.tensor([1]))
    print(log_p, v)
    a, v = model.step(img)
    print(a,v)
    a = model.act(img)
    print(a)
    




