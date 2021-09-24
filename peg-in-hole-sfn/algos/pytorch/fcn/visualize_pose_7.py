import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.optim import Adam
import gym
import time
from utils.logx import EpochLogger
from algos.pytorch.fcn.unet import UNet
import matplotlib.pyplot as plt
import random
from PIL import Image
import copy
 
import numpy as np
np.set_printoptions(threshold=np.inf)
 
 
 
class Buffer(object):
    def __init__(self, buffer_size, rot_num=18):
        super(Buffer, self).__init__()
        self.anchor_buf = np.zeros(shape=(buffer_size, 200, 250), dtype=np.float32)
        self.pos_buf = np.zeros(shape=(buffer_size, 200, 250), dtype=np.float32)
        self.neg_buf = np.zeros(shape=(buffer_size, rot_num, 200, 250), dtype=np.float32)
        self.ptr = 0

    def store(self, anchor, pos, neg):
        self.anchor_buf[self.ptr] = anchor
        self.pos_buf[self.ptr] = pos
        self.neg_buf[self.ptr] = neg
        self.ptr += 1

    def get(self):
        self.ptr = 0
        data = dict(anchor=self.anchor_buf,
                    pos=self.pos_buf,
                    neg=self.neg_buf)
        return data



def test_pose(venv, nenv, seed=0, local_steps_per_epoch=1000, 
    device='cpu', model_path='', rot_num=4):
    # Random seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Load model
    model = torch.load(model_path, map_location=device)

    # Set up experience buffer
    buf_list = [Buffer(local_steps_per_epoch, rot_num) for _ in range(nenv)]

    optimizer = Adam(model.parameters(), lr=1e-4)


    # Main loop: collect experience in env and update/log each epoch
    o = venv.reset()
    # collecting data
    for t in range(local_steps_per_epoch):

        [buf_list[i].store(o['anchor'][i], o['pos'][i], o['neg'][i]) for i in range(nenv)]

        # fake action
        o, _, _, _ = venv.step([5]*nenv)

    anchor = torch.cat([torch.tensor(
        buf_list[i].get()['anchor']) for i in range(nenv)])
    pos = torch.cat([torch.tensor(
        buf_list[i].get()['pos']) for i in range(nenv)])
    neg = torch.cat([torch.tensor(
        buf_list[i].get()['neg']) for i in range(nenv)])


    # for i in range(10):
    #     plt.subplot(3,4,1)
    #     plt.imshow(seg_pred[0+i], cmap=plt.cm.gray)
    #     plt.subplot(3,4,2)
    #     plt.imshow(seg_gt[0+i], cmap=plt.cm.gray)
    #     plt.subplot(3,4,3)
    #     plt.imshow(anchor[0+i], cmap=plt.cm.gray)
    #     plt.subplot(3,4,4)
    #     plt.imshow(pos[0+i], cmap=plt.cm.gray)
    #     plt.subplot(3,4,5)
    #     plt.imshow(neg[0+i][0], cmap=plt.cm.gray)
    #     plt.subplot(3,4,6)
    #     plt.imshow(neg[0+i][1], cmap=plt.cm.gray)
    #     plt.subplot(3,4,7)
    #     plt.imshow(neg[0+i][2], cmap=plt.cm.gray)
    #     plt.subplot(3,4,8)
    #     plt.imshow(neg[0+i][3], cmap=plt.cm.gray)
    #     plt.subplot(3,4,9)
    #     plt.imshow(neg[0+i][4], cmap=plt.cm.gray)
    #     plt.subplot(3,4,10)
    #     plt.imshow(neg[0+i][5], cmap=plt.cm.gray)
    #     plt.show()
    # exit(0)

    acc = 0
    # evaluating network
    with torch.no_grad():
        for idx in range(anchor.size(0)):
            anchor_eval = anchor[idx:idx+1].to(device)
            pos_eval = pos[idx:idx+1].to(device)
            neg_eval = neg[idx].to(device)

            inputs = torch.cat((anchor_eval, pos_eval, neg_eval), dim=0)
            outs = model(inputs.unsqueeze(1))
            anchor_flatten = outs[0].view(1,-1)
            pos_flatten = outs[1].view(1,-1)
            neg_flatten = outs[2:].view(rot_num,-1)

            pos_dis = F.pairwise_distance(anchor_flatten, pos_flatten)
            neg_dis = F.pairwise_distance(anchor_flatten.repeat(rot_num,1), neg_flatten)
            neg_dis_clamp = torch.clamp(1-neg_dis, min=0).mean()
            loss = pos_dis + neg_dis_clamp

            contrasAcc = 0 if neg_dis.le(pos_dis).any() else 1

            acc += contrasAcc
            print(idx, contrasAcc, loss)

    print('contrasAcc', acc/anchor.size(0))


 

if __name__ == '__main__':
    pass
