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
import cv2
import numpy as np
import math
np.set_printoptions(threshold=np.inf)
from utils.pck_acc import accuracy



def _fast_hist(label_true, label_pred, n_class):
    mask = (label_true >= 0) & (label_true < n_class)
    hist = np.bincount(
        n_class * label_true[mask].astype(int) +
        label_pred[mask], minlength=n_class ** 2).reshape(n_class, n_class)
    return hist


def label_accuracy_score(label_trues, label_preds, n_class):
    """Returns accuracy score evaluation result as defined in FCN

      - overall accuracy
      - mean accuracy
      - mean IU
      - fwavacc
    """
    hist = np.zeros((n_class, n_class))
    for lt, lp in zip(label_trues, label_preds):
        hist += _fast_hist(lt.flatten(), lp.flatten(), n_class)
    acc = np.diag(hist).sum() / hist.sum()
    with np.errstate(divide='ignore', invalid='ignore'):
        acc_cls = np.diag(hist) / hist.sum(axis=1)
    acc_cls = np.nanmean(acc_cls)
    with np.errstate(divide='ignore', invalid='ignore'):
        iu = np.diag(hist) / (
            hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist)
        )
    mean_iu = np.nanmean(iu)
    freq = hist.sum(axis=1) / hist.sum()
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
    return acc, acc_cls, mean_iu, fwavacc



 
class Buffer(object):
    def __init__(self, buffer_size):
        super(Buffer, self).__init__()
        self.seg_pred_buf = np.zeros(shape=(buffer_size, 200, 250), dtype=np.float32)
        self.seg_gt_buf = np.zeros(shape=(buffer_size, 200, 250), dtype=np.int32)
        self.peg_gt_buf = np.zeros(shape=(buffer_size, 200, 250), dtype=np.float32)
        self.hole_gt_buf = np.zeros(shape=(buffer_size, 200, 250), dtype=np.float32)
        self.ptr = 0

    def store(self, seg_pred, seg_gt, peg_gt, hole_gt):
        self.seg_pred_buf[self.ptr] = seg_pred
        self.seg_gt_buf[self.ptr] = seg_gt
        self.peg_gt_buf[self.ptr] = peg_gt
        self.hole_gt_buf[self.ptr] = hole_gt
        self.ptr += 1

    def get(self):
        self.ptr = 0
        data = dict(seg_pred=self.seg_pred_buf, 
                    seg_gt=self.seg_gt_buf,
                    peg_gt=self.peg_gt_buf,
                    hole_gt=self.hole_gt_buf)
        return data


def transform(o):
    img = o['img']
    gt_mask = o['gt']
    peg_xy = o['peg_xy']
    hole_xy = o['hole_xy']
    with torch.no_grad():
        # img = torch.from_numpy(img.astype(np.float32)).to(device)
        # seg_pred = seg_model(img).max(1)[1].cpu().numpy()
        seg_pred = gt_mask

    # get peg_gt and hole_gt
    peg_gt = np.zeros_like(gt_mask, dtype=np.float32)
    sigma = 3
    n,h,w = peg_gt.shape
    for idx in range(n):
        x_min = int(max(0, peg_xy[idx][0]-sigma))
        x_max = int(min(w-1, peg_xy[idx][0]+sigma))
        y_min = int(max(0, peg_xy[idx][1]-sigma))
        y_max = int(min(h-1, peg_xy[idx][1]+sigma))
        for i in range(x_min,x_max):
            for j in range(y_min,y_max):
                dx = peg_xy[idx][0]-i
                dy = peg_xy[idx][1]-j
                v = math.exp(-(pow(dx,2)+pow(dy,2))/(2*pow(sigma,2)))
                if v < 0.7:
                    continue
                peg_gt[idx][j][i] = v

    hole_gt = np.zeros_like(gt_mask, dtype=np.float32)
    sigma = 3
    n,h,w = hole_gt.shape
    for idx in range(n):
        x_min = int(max(0, hole_xy[idx][0]-sigma))
        x_max = int(min(w-1, hole_xy[idx][0]+sigma))
        y_min = int(max(0, hole_xy[idx][1]-sigma))
        y_max = int(min(h-1, hole_xy[idx][1]+sigma))
        for i in range(x_min,x_max):
            for j in range(y_min,y_max):
                dx = hole_xy[idx][0]-i
                dy = hole_xy[idx][1]-j
                v = math.exp(-(pow(dx,2)+pow(dy,2))/(2*pow(sigma,2)))
                if v < 0.7:
                    continue
                hole_gt[idx][j][i] = v

    return seg_pred, peg_gt, hole_gt
 

def test_position(venv, nenv, seed=0, local_steps_per_epoch=1000, 
    device='cpu', model_path=''):
    # Random seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Load model
    model = torch.load(model_path, map_location=device)
    criterion = nn.MSELoss(reduction='mean')

    # Set up experience buffer
    buf_list = [Buffer(local_steps_per_epoch) for _ in range(nenv)]


    # Main loop: collect experience in env and update/log each epoch
    o = venv.reset()
    # collecting data
    for t in range(local_steps_per_epoch):
        seg_pred, peg_gt, hole_gt = transform(o)

        [buf_list[i].store(
            seg_pred[i], 
            o['gt'][i],
            peg_gt[i],
            hole_gt[i]) for i in range(nenv)]

        # fake action
        o, _, _, _ = venv.step([20]*nenv)

    seg_pred = torch.cat([torch.tensor(
        buf_list[i].get()['seg_pred']) for i in range(nenv)])
    seg_gt = torch.cat([torch.tensor(
        buf_list[i].get()['seg_gt']) for i in range(nenv)])
    peg_gt = torch.cat([torch.tensor(
        buf_list[i].get()['peg_gt']) for i in range(nenv)])
    hole_gt = torch.cat([torch.tensor(
        buf_list[i].get()['hole_gt']) for i in range(nenv)])


    peg_acc_10 = 0
    hole_acc_10 = 0
    peg_acc_20 = 0
    hole_acc_20 = 0
    loss_avg = 0
    ls_test = list(range(local_steps_per_epoch))
    # evaluating network
    with torch.no_grad():
        for idx in ls_test:
            seg_pred_batch = seg_pred[idx:idx+1].to(device)
            seg_gt_batch = seg_gt[idx:idx+1]
            peg_gt_batch = peg_gt[idx:idx+1].to(device)
            hole_gt_batch = hole_gt[idx:idx+1].to(device)

            outs = model(seg_pred_batch.unsqueeze(1))
            loss_peg = 0.5 * criterion(outs[:,0,:,:], peg_gt_batch)
            loss_hole = 0.5 * criterion(outs[:,1,:,:], hole_gt_batch)
            loss = loss_peg + loss_hole

            _, acc_position_peg_10, _, _ = accuracy(
                outs[:,0:1,:,:].detach().cpu().numpy(),
                peg_gt_batch.unsqueeze(1).detach().cpu().numpy(), thr=10)

            _, acc_position_hole_10, _, _ = accuracy(
                outs[:,1:2,:,:].detach().cpu().numpy(),
                hole_gt_batch.unsqueeze(1).detach().cpu().numpy(), thr=10)

            _, acc_position_peg_20, _, _ = accuracy(
                outs[:,0:1,:,:].detach().cpu().numpy(),
                peg_gt_batch.unsqueeze(1).detach().cpu().numpy(), thr=20)

            _, acc_position_hole_20, _, _ = accuracy(
                outs[:,1:2,:,:].detach().cpu().numpy(),
                hole_gt_batch.unsqueeze(1).detach().cpu().numpy(), thr=20)

            _, _, mean_iu, _ = label_accuracy_score(
                seg_gt_batch.numpy(), seg_pred_batch.int().detach().cpu().numpy(), 3)

            loss_avg += loss.item()
            peg_acc_10 += acc_position_peg_10
            hole_acc_10 += acc_position_hole_10
            peg_acc_20 += acc_position_peg_20
            hole_acc_20 += acc_position_hole_20

    print('loss', loss_avg/local_steps_per_epoch)
    print('peg_acc_10', peg_acc_10/local_steps_per_epoch)
    print('hole_acc_10', hole_acc_10/local_steps_per_epoch)
    print('peg_acc_20', peg_acc_20/local_steps_per_epoch)
    print('hole_acc_20', hole_acc_20/local_steps_per_epoch)


 

if __name__ == '__main__':
    pass
