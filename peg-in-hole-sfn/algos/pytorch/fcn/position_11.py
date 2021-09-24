import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.optim import Adam
import gym
import time
import matplotlib.pyplot as plt
import random
from PIL import Image
import copy
import cv2
import math

from utils.pck_acc import accuracy
from utils.logx import EpochLogger
from utils.utils import get_position_gt, label_accuracy_score, Buffer
from algos.pytorch.fcn.unet_11 import UNet



def train_position(venv, nenv, seed=0, local_steps_per_epoch=1000, 
    epochs=50, logger_kwargs=dict(), save_freq=10, device='cpu', 
    resume=False, position_model_path='', seg_model_path='',
    iterates=100, batch_size=10):

    logger = EpochLogger(**logger_kwargs)

    # Random seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Load model
    if resume:
        position_model = torch.load(position_model_path, map_location=device)
    else:
        position_model = UNet(1,2).to(device)
    # seg_model = torch.load(seg_model_path, map_location=device)
    buf_list = [Buffer(local_steps_per_epoch) for _ in range(nenv)]
    optimizer = Adam(position_model.parameters(), lr=1e-4)
    # criterion = nn.MSELoss(reduction='mean')
    criterion = nn.CrossEntropyLoss()
    logger.setup_pytorch_saver(position_model)
    
    start_time = time.time()

    # Main loop: collect experience in env and update/log each epoch
    for ite in range(iterates):
        o = venv.reset()
        # collecting data
        for t in range(local_steps_per_epoch):
            position_gts = get_position_gt(o)

            seg_pred = o['gt']
            [buf_list[i].store(
                seg_pred=seg_pred[i], 
                seg_gt=o['gt'][i], 
                position_gt=position_gts[i]) for i in range(nenv)]


            # plt.subplot(4,4,1)
            # plt.imshow(o['gt'][0])
            # plt.subplot(4,4,2)
            # plt.imshow(position_gts[0])
            # plt.subplot(4,4,3)
            # plt.imshow(o['gt'][1])
            # plt.subplot(4,4,4)
            # plt.imshow(position_gts[1])
            # plt.subplot(4,4,5)
            # plt.imshow(o['gt'][2])
            # plt.subplot(4,4,6)
            # plt.imshow(position_gts[2])
            # plt.subplot(4,4,7)
            # plt.imshow(o['gt'][3])
            # plt.subplot(4,4,8)
            # plt.imshow(position_gts[3])
            # plt.subplot(4,4,9)
            # plt.imshow(o['gt'][4])
            # plt.subplot(4,4,10)
            # plt.imshow(position_gts[4])
            # plt.show()

            o, _, _, _ = venv.step([20]*nenv)



        seg_pred = torch.cat([torch.tensor(
            buf_list[i].get()['seg_pred']) for i in range(nenv)])
        seg_gt = torch.cat([torch.tensor(
            buf_list[i].get()['seg_gt']) for i in range(nenv)])
        position_gt = torch.cat([torch.tensor(
            buf_list[i].get()['position_gt'], dtype=torch.long) for i in range(nenv)])


        # split training and testing dataset
        ls_train = list(range((nenv-1)*local_steps_per_epoch))
        ls_test = list(range((nenv-1)*local_steps_per_epoch, nenv*local_steps_per_epoch))

        for epoch in range(epochs):
            # training network
     #        random.shuffle(ls_train)
     #        for i, idx in enumerate(range(0, len(ls_train), batch_size)):
     #            # batch shape: [N, 224, 224]
     #            seg_pred_batch = seg_pred[ls_train[idx:idx+batch_size]].to(device)
     #            seg_gt_batch = seg_gt[ls_train[idx:idx+batch_size]]
     #            position_gt_batch = position_gt[ls_train[idx:idx+batch_size]].to(device)

     #            optimizer.zero_grad()
     #            outs = position_model(seg_pred_batch.unsqueeze(1))

     #            n,c,h,w = outs.size()
     #            outs_flatten = outs.permute(0,2,3,1).contiguous().view(-1,c)
     #            gts_flatten = position_gt_batch.view(-1)

     #            loss = criterion(outs_flatten, gts_flatten)
     #            loss.backward()
     #            optimizer.step()
     #            # !!!!!!!!!!!!!!!
     #            preds = outs.max(1)[1]
     #            # preds = (outs[:,1,:,:].max()==outs[:,1,:,:]).type(torch.long)

     #            _, acc_position_1, _, _ = accuracy(
     #                preds.unsqueeze(1).detach().cpu().numpy(),
     #                position_gt_batch.unsqueeze(1).detach().cpu().numpy(), thr=1)

     #            _, acc_position_5, _, _ = accuracy(
					# preds.unsqueeze(1).detach().cpu().numpy(),
					# position_gt_batch.unsqueeze(1).detach().cpu().numpy(), thr=5)

     #            _, acc_position_10, _, _ = accuracy(
     #                preds.unsqueeze(1).detach().cpu().numpy(),
     #                position_gt_batch.unsqueeze(1).detach().cpu().numpy(), thr=10)

     #            _, acc_position_20, _, _ = accuracy(
     #                preds.unsqueeze(1).detach().cpu().numpy(),
     #                position_gt_batch.unsqueeze(1).detach().cpu().numpy(), thr=20)

     #            _, _, mean_iu, _ = label_accuracy_score(
     #                seg_gt_batch.numpy(), seg_pred_batch.int().detach().cpu().numpy(), 3)

     #            logger.store(trainLoss=loss.item())
     #            logger.store(trainMeanIU=mean_iu.item())
     #            logger.store(trainPositionAcc_1=acc_position_1)
     #            logger.store(trainPositionAcc_5=acc_position_5)
     #            logger.store(trainPositionAcc_10=acc_position_10)
     #            logger.store(trainPositionAcc_20=acc_position_20)

            # evaluating network
            with torch.no_grad():
                for idx in ls_test:
                    seg_pred_batch = seg_pred[idx:idx+1].to(device)
                    seg_gt_batch = seg_gt[idx:idx+1]
                    position_gt_batch = position_gt[idx:idx+1].to(device)

                    outs = position_model(seg_pred_batch.unsqueeze(1))
                    n,c,h,w = outs.size()
                    outs_flatten = outs.permute(0,2,3,1).contiguous().view(-1,c)
                    gts_flatten = position_gt_batch.view(-1)
                    loss = criterion(outs_flatten, gts_flatten)
                    # preds = outs.max(1)[1]
                    # !!!!!!!!!!!
                    preds = (outs[:,1,:,:].max()==outs[:,1,:,:]).type(torch.long)
                    # plt.subplot(1,2,1)
                    # plt.imshow(position_gt_batch[0].detach().cpu().numpy())
                    # plt.subplot(1,2,2)
                    # plt.imshow(preds[0].detach().cpu().numpy())
                    # plt.show()

                    # plt.subplot(2,3,1)
                    # plt.imshow(seg_pred_batch[0].detach().cpu().numpy())
                    # plt.subplot(2,3,2)
                    # plt.imshow(position_gt_batch[0].detach().cpu().numpy())
                    # plt.subplot(2,3,3)
                    # plt.imshow(preds[0].detach().cpu().numpy())
                    # plt.subplot(2,3,4)
                    # plt.imshow(outs[0][0].detach().cpu().numpy())
                    # plt.subplot(2,3,5)
                    # plt.imshow(outs[0][1].detach().cpu().numpy())
                    # # plt.imshow(2,3,6)
                    # # plt.imshow()
                    # plt.show()

                    _, acc_position_1, _, _ = accuracy(
                        preds.unsqueeze(1).detach().cpu().numpy(),
                        position_gt_batch.unsqueeze(1).detach().cpu().numpy(), thr=1)

                    _, acc_position_5, _, _ = accuracy(
                        preds.unsqueeze(1).detach().cpu().numpy(),
                        position_gt_batch.unsqueeze(1).detach().cpu().numpy(), thr=5)

                    _, acc_position_10, _, _ = accuracy(
                        preds.unsqueeze(1).detach().cpu().numpy(),
                        position_gt_batch.unsqueeze(1).detach().cpu().numpy(), thr=10)

                    _, acc_position_20, _, _ = accuracy(
                        preds.unsqueeze(1).detach().cpu().numpy(),
                        position_gt_batch.unsqueeze(1).detach().cpu().numpy(), thr=20)

                    _, _, mean_iu, _ = label_accuracy_score(
                        seg_gt_batch.numpy(), seg_pred_batch.int().detach().cpu().numpy(), 3)


                    logger.store(evalLoss=loss.item())
                    logger.store(evalMeanIU=mean_iu.item())
                    logger.store(evalPositionAcc_1=acc_position_1)
                    logger.store(evalPositionAcc_5=acc_position_5)
                    logger.store(evalPositionAcc_10=acc_position_10)
                    logger.store(evalPositionAcc_20=acc_position_20)


            # Log info about epoch
            logger.log_tabular('Iterates', ite)
            logger.log_tabular('Epoch', epoch)
            # logger.log_tabular('trainLoss', average_only=True)
            # logger.log_tabular('trainMeanIU', average_only=True)
            # logger.log_tabular('trainPositionAcc_1', average_only=True)
            # logger.log_tabular('trainPositionAcc_5', average_only=True)
            # logger.log_tabular('trainPositionAcc_10', average_only=True)
            # logger.log_tabular('trainPositionAcc_20', average_only=True)

            logger.log_tabular('evalLoss', average_only=True)
            logger.log_tabular('evalMeanIU', average_only=True)
            logger.log_tabular('evalPositionAcc_1', average_only=True)
            logger.log_tabular('evalPositionAcc_5', average_only=True)
            logger.log_tabular('evalPositionAcc_10', average_only=True)
            logger.log_tabular('evalPositionAcc_20', average_only=True)

            logger.log_tabular('TotalEnvInteracts', (epoch+1)*local_steps_per_epoch*nenv)
            logger.log_tabular('Time', time.time()-start_time)
            logger.dump_tabular()

            # # Save model
            # if (epoch % save_freq == 0) or (epoch == epochs-1):
            #     logger.save_state({}, None)
            #     print('model saved !')



if __name__ == '__main__':
    pass
