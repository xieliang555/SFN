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
import math
   
import numpy as np
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



def train_position(venv, nenv, seed=0, local_steps_per_epoch=1000, 
    epochs=50, logger_kwargs=dict(), save_freq=10, device='cpu', 
    resume=False, model_path='', iterates=100, batch_size=1, 
    seg_model_path=''):

    logger = EpochLogger(**logger_kwargs)
    # logger.save_config(locals())

    # Random seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Load model
    if resume:
        model = torch.load(model_path, map_location=device)
    else:
        model = UNet(1,2).to(device)
    # seg_model = torch.load(seg_model_path, map_location=device)

    # Set up experience buffer
    buf_list = [Buffer(local_steps_per_epoch) for _ in range(nenv)]

    optimizer = Adam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss(reduction='mean')

    # Set up model saving
    logger.setup_pytorch_saver(model)

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

        # plt.subplot(2,3,1)
        # plt.imshow(gt_mask[0])
        # plt.subplot(2,3,2)
        # plt.imshow(peg_gt[0])
        # plt.subplot(2,3,3)
        # plt.imshow(cv2.addWeighted(gt_mask[0].astype(np.float32),1, peg_gt[0],1, 0))
        # plt.subplot(2,3,4)
        # plt.imshow(hole_gt[0])
        # plt.subplot(2,3,5)
        # plt.imshow(cv2.addWeighted(gt_mask[0].astype(np.float32),1, hole_gt[0],1, 0))
        # plt.show()

        return seg_pred, peg_gt, hole_gt

    start_time = time.time()

    # Main loop: collect experience in env and update/log each epoch
    for ite in range(iterates):
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



        # split training and testing dataset
        ls_train = list(range((nenv-1)*local_steps_per_epoch))
        ls_test = list(range((nenv-1)*local_steps_per_epoch, nenv*local_steps_per_epoch))

        for epoch in range(epochs):
            # training network
            random.shuffle(ls_train)
            for i, idx in enumerate(range(0, len(ls_train), batch_size)):
                # batch shape: [N, 224, 224]
                seg_pred_batch = seg_pred[ls_train[idx:idx+batch_size]].to(device)
                seg_gt_batch = seg_gt[ls_train[idx:idx+batch_size]]
                peg_gt_batch = peg_gt[ls_train[idx:idx+batch_size]].to(device)
                hole_gt_batch = hole_gt[ls_train[idx:idx+batch_size]].to(device)

                optimizer.zero_grad()
                outs = model(seg_pred_batch.unsqueeze(1))

                loss_peg = 0.5 * criterion(outs[:,0,:,:], peg_gt_batch)
                loss_hole = 0.5 * criterion(outs[:,1,:,:], hole_gt_batch)
                loss = loss_peg + loss_hole

                loss.backward()
                optimizer.step()


                _, acc_position_peg_1, _, _ = accuracy(
                    outs[:,0:1,:,:].detach().cpu().numpy(),
                    peg_gt_batch.unsqueeze(1).detach().cpu().numpy(), thr=1)

                _, acc_position_hole_1, _, _ = accuracy(
                    outs[:,1:2,:,:].detach().cpu().numpy(),
                    hole_gt_batch.unsqueeze(1).detach().cpu().numpy(), thr=1)

                _, acc_position_peg_3, _, _ = accuracy(
                    outs[:,0:1,:,:].detach().cpu().numpy(),
                    peg_gt_batch.unsqueeze(1).detach().cpu().numpy(), thr=3)

                _, acc_position_hole_3, _, _ = accuracy(
                    outs[:,1:2,:,:].detach().cpu().numpy(),
                    hole_gt_batch.unsqueeze(1).detach().cpu().numpy(), thr=3)

                _, _, mean_iu, _ = label_accuracy_score(
                    seg_gt_batch.numpy(), seg_pred_batch.int().detach().cpu().numpy(), 3)

                logger.store(trainLoss=loss.item())
                logger.store(trainMeanIU=mean_iu.item())
                logger.store(trainPegPositionAcc_1=acc_position_peg_1)
                logger.store(trainHolePositionAcc_1=acc_position_hole_1)
                logger.store(trainPegPositionAcc_3=acc_position_peg_3)
                logger.store(trainHolePositionAcc_3=acc_position_hole_3)


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

                    _, acc_position_peg_1, _, _ = accuracy(
                        outs[:,0:1,:,:].detach().cpu().numpy(),
                        peg_gt_batch.unsqueeze(1).detach().cpu().numpy(), thr=1)

                    _, acc_position_hole_1, _, _ = accuracy(
                        outs[:,1:2,:,:].detach().cpu().numpy(),
                        hole_gt_batch.unsqueeze(1).detach().cpu().numpy(), thr=1)

                    _, acc_position_peg_3, _, _ = accuracy(
                        outs[:,0:1,:,:].detach().cpu().numpy(),
                        peg_gt_batch.unsqueeze(1).detach().cpu().numpy(), thr=3)

                    _, acc_position_hole_3, _, _ = accuracy(
                        outs[:,1:2,:,:].detach().cpu().numpy(),
                        hole_gt_batch.unsqueeze(1).detach().cpu().numpy(), thr=3)

                    _, _, mean_iu, _ = label_accuracy_score(
                        seg_gt_batch.numpy(), seg_pred_batch.int().detach().cpu().numpy(), 3)


                    logger.store(evalLoss=loss.item())
                    logger.store(evalMeanIU=mean_iu.item())
                    logger.store(evalPegPositionAcc_1=acc_position_peg_1)
                    logger.store(evalHolePositionAcc_1=acc_position_hole_1)
                    logger.store(evalPegPositionAcc_3=acc_position_peg_3)
                    logger.store(evalHolePositionAcc_3=acc_position_hole_3)


            # Log info about epoch
            logger.log_tabular('Iterates', ite)
            logger.log_tabular('Epoch', epoch)
            logger.log_tabular('trainLoss', average_only=True)
            logger.log_tabular('trainMeanIU', average_only=True)
            logger.log_tabular('trainPegPositionAcc_1', average_only=True)
            logger.log_tabular('trainHolePositionAcc_1', average_only=True)
            logger.log_tabular('trainPegPositionAcc_3', average_only=True)
            logger.log_tabular('trainHolePositionAcc_3', average_only=True)

            logger.log_tabular('evalLoss', average_only=True)
            logger.log_tabular('evalMeanIU', average_only=True)
            logger.log_tabular('evalPegPositionAcc_1', average_only=True)
            logger.log_tabular('evalHolePositionAcc_1', average_only=True)
            logger.log_tabular('evalPegPositionAcc_3', average_only=True)
            logger.log_tabular('evalHolePositionAcc_3', average_only=True)

            logger.log_tabular('TotalEnvInteracts', (epoch+1)*local_steps_per_epoch*nenv)
            logger.log_tabular('Time', time.time()-start_time)
            logger.dump_tabular()

            # Save model
            if (epoch % save_freq == 0) or (epoch == epochs-1):
                logger.save_state({}, None)
                print('model saved !')



if __name__ == '__main__':
    pass
