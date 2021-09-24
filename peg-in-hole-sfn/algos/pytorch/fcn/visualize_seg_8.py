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
 
import numpy as np
np.set_printoptions(threshold=np.inf)
 
 
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
        self.obs_img_buf = np.zeros(shape=(buffer_size, 3, 400, 500), dtype=np.float32)
        self.gt_buf = np.zeros(shape=(buffer_size, 400, 500), dtype=np.float32)
        self.ptr = 0

    def store(self, obs_img, gt):
        self.obs_img_buf[self.ptr] = obs_img
        self.gt_buf[self.ptr] = gt
        self.ptr += 1

    def get(self):
        self.ptr = 0
        data = dict(obs_img=self.obs_img_buf,
                    gt=self.gt_buf)
        return data

 

def test_seg(venv, nenv, seed=0, steps=1000, device='cpu', model_path=''):

    # Random seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Load model
    model = torch.load(model_path).to(device)
    buf_list = [Buffer(steps) for _ in range(nenv)]
    criterion = nn.CrossEntropyLoss()

    o = venv.reset()
    # collecting data
    for t in range(steps):
        [buf_list[i].store(o['img'][i], o['gt'][i]) for i in range(nenv)]
        # fake action
        o, _, _, _ = venv.step([0]*nenv)

    obs_img = torch.cat([torch.tensor(
        buf_list[i].get()['obs_img']) for i in range(nenv)])
    gt = torch.cat([torch.tensor(
        buf_list[i].get()['gt'], dtype=torch.long) for i in range(nenv)])


    # evaluating
    mean_iu_eval = 0
    with torch.no_grad():
        for i in range(0, len(obs_img)):
            img_eval = obs_img[i:i+1].to(device)
            gt_eval = gt[i:i+1].to(device)
            
            preds = model(img_eval)

            n,c,h,w = preds.size()
            preds_flatten = preds.permute(0,2,3,1).contiguous().view(-1,c)
            gts_flatten = gt_eval.view(-1)

            loss = criterion(preds_flatten, gts_flatten)

            preds_batch = preds.max(1)[1].cpu().numpy()
            gt_batch = gt_eval.cpu().numpy()
            acc, acc_cls, mean_iu, fwavacc = label_accuracy_score(
                gt_batch, preds_batch, 3)

            mean_iu_eval += mean_iu

            plt.subplot(1,3,1)
            plt.imshow(img_eval[0].cpu().detach().permute(1,2,0)/255)
            plt.subplot(1,3,2)
            plt.imshow(preds_batch[0])
            plt.subplot(1,3,3)
            plt.imshow(gt_batch[0])
            plt.show()

        mean_iu_eval /= len(obs_img)
        print('mean_iu_eval', mean_iu_eval)





if __name__ == '__main__':
    pass
