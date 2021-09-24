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
    def __init__(self, buffer_size, rot_num=6):
        super(Buffer, self).__init__()
        self.seg_pred_buf = np.zeros(shape=(buffer_size, 200, 250), dtype=np.int32)
        self.seg_gt_buf = np.zeros(shape=(buffer_size, 200, 250), dtype=np.int32)
        self.anchor_buf = np.zeros(shape=(buffer_size, 200, 250), dtype=np.float32)
        self.pos_buf = np.zeros(shape=(buffer_size, 200, 250), dtype=np.float32)
        self.neg_buf = np.zeros(shape=(buffer_size, rot_num, 200, 250), dtype=np.float32)
        self.ptr = 0

    def store(self, seg_pred, seg_gt, anchor, pos, neg):
        self.seg_pred_buf[self.ptr] = seg_pred
        self.seg_gt_buf[self.ptr] = seg_gt
        self.anchor_buf[self.ptr] = anchor
        self.pos_buf[self.ptr] = pos
        self.neg_buf[self.ptr] = neg
        self.ptr += 1

    def get(self):
        self.ptr = 0
        data = dict(seg_pred=self.seg_pred_buf, 
                    seg_gt=self.seg_gt_buf,
                    anchor=self.anchor_buf,
                    pos=self.pos_buf,
                    neg=self.neg_buf)
        return data



def train_pose(venv, nenv, seed=0, local_steps_per_epoch=1000, 
    epochs=50, logger_kwargs=dict(), save_freq=10, device='cpu', 
    resume=False, model_path='', iterates=100, batch_size=1, 
    seg_model_path='', rot_num=10):

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
        model = UNet(1,64).to(device)
    # seg_model = torch.load(seg_model_path, map_location=device)

    # Set up experience buffer
    buf_list = [Buffer(local_steps_per_epoch, rot_num) for _ in range(nenv)]

    optimizer = Adam(model.parameters(), lr=1e-4)


    # Set up model saving
    logger.setup_pytorch_saver(model)

    def transform(o):
        '''
        transform rgb image to anchor, positive and negative sample pair.
        '''
        img = o['img']
        gt_mask = o['gt']
        dtheta = o['dyaw']
        # print(img.shape) [5,3,224,224]
        with torch.no_grad():
            # img = torch.from_numpy(img.astype(np.float32)).to(device)
            # seg_pred = seg_model(img).max(1)[1].cpu().numpy()
            seg_pred = gt_mask
            peg_mask = np.uint8(seg_pred==1)
            base_mask = np.uint8(seg_pred==2)

        # transform 
        rotation_ls = np.array([-10,-8,-6,-4,-2, 0, 2,4,6,8,10])
        idx_pos = [abs(i-rotation_ls).argmin() for i in dtheta]
        theta_pos = rotation_ls[idx_pos]
        idx_neg = [[i for i in range(len(rotation_ls)) if i != j] for j in idx_pos]
        theta_neg = np.array([rotation_ls[i] for i in idx_neg])
        base_mask_pos = [Image.fromarray(base_mask[i]).rotate(
            theta_pos[i], expand=False, fillcolor=0) for i in range(len(dtheta))]
        base_mask_neg = [[np.array(Image.fromarray(base_mask[i]).rotate(t, 
            expand=False, fillcolor=0)) for t in theta_neg[i]] for i in range(len(dtheta))]
        
        # for i in range(5):
        #     ax1 = plt.subplot(3,4,1)
        #     ax1.set_title(str(theta_pos[i])+'/'+str(dtheta[i]))
        #     plt.imshow(cv2.addWeighted(np.array(base_mask_pos[i]), 1,peg_mask[i], 2, 0))
        #     ax2 = plt.subplot(3,4,2)
        #     ax2.set_title(str(theta_neg[i][0]))
        #     plt.imshow(cv2.addWeighted(np.array(base_mask_neg[i][0]), 1, peg_mask[i], 2, 0))
        #     ax3 = plt.subplot(3,4,3)
        #     ax3.set_title(str(theta_neg[i][1]))
        #     plt.imshow(cv2.addWeighted(np.array(base_mask_neg[i][1]), 1, peg_mask[i], 2, 0))
        #     ax4 = plt.subplot(3,4,4)
        #     ax4.set_title(str(theta_neg[i][2]))
        #     plt.imshow(cv2.addWeighted(np.array(base_mask_neg[i][2]), 1, peg_mask[i], 2, 0))
        #     ax5 = plt.subplot(3,4,5)
        #     ax5.set_title(str(theta_neg[i][3]))
        #     plt.imshow(cv2.addWeighted(np.array(base_mask_neg[i][3]), 1, peg_mask[i], 2, 0))
        #     ax6 = plt.subplot(3,4,6)
        #     ax6.set_title(str(theta_neg[i][4]))
        #     plt.imshow(cv2.addWeighted(np.array(base_mask_neg[i][4]), 1, peg_mask[i], 2, 0))
        #     ax7 = plt.subplot(3,4,7)
        #     ax7.set_title(str(theta_neg[i][5]))
        #     plt.imshow(cv2.addWeighted(np.array(base_mask_neg[i][5]), 1, peg_mask[i], 2, 0))
        #     ax8 = plt.subplot(3,4,8)
        #     ax8.set_title(str(theta_neg[i][6]))
        #     plt.imshow(cv2.addWeighted(np.array(base_mask_neg[i][6]), 1, peg_mask[i], 2, 0))
        #     ax9 = plt.subplot(3,4,9)
        #     ax9.set_title(str(theta_neg[i][7]))
        #     plt.imshow(cv2.addWeighted(np.array(base_mask_neg[i][7]), 1, peg_mask[i], 2, 0))
        #     ax10 = plt.subplot(3,4,10)
        #     ax10.set_title(str(theta_neg[i][8]))
        #     plt.imshow(cv2.addWeighted(np.array(base_mask_neg[i][8]), 1, peg_mask[i], 2, 0))
        #     ax10 = plt.subplot(3,4,11)
        #     ax10.set_title(str(theta_neg[i][9]))
        #     plt.imshow(cv2.addWeighted(np.array(base_mask_neg[i][9]), 1, peg_mask[i], 2, 0))
        #     plt.subplot(3,4,12)
        #     plt.imshow(img[i].transpose((1,2,0)))
        #     plt.show()

        return seg_pred, peg_mask, base_mask_pos, base_mask_neg

    start_time = time.time()

    # Main loop: collect experience in env and update/log each epoch
    for ite in range(iterates):
        o = venv.reset()
        # collecting data
        for t in range(local_steps_per_epoch):
            seg_pred, peg_mask, base_mask_pos, base_mask_neg = transform(o)

            [buf_list[i].store(
                seg_pred[i], 
                o['gt'][i],
                peg_mask[i],
                base_mask_pos[i],
                base_mask_neg[i]) for i in range(nenv)]

            # fake action
            o, _, _, _ = venv.step([20]*nenv)

        seg_pred = torch.cat([torch.tensor(
            buf_list[i].get()['seg_pred']) for i in range(nenv)])
        seg_gt = torch.cat([torch.tensor(
            buf_list[i].get()['seg_gt']) for i in range(nenv)])
        anchor = torch.cat([torch.tensor(
            buf_list[i].get()['anchor']) for i in range(nenv)])
        pos = torch.cat([torch.tensor(
            buf_list[i].get()['pos']) for i in range(nenv)])
        neg = torch.cat([torch.tensor(
            buf_list[i].get()['neg']) for i in range(nenv)])



        # split training and testing dataset
        ls_train = list(range((nenv-1)*local_steps_per_epoch))
        ls_test = list(range((nenv-1)*local_steps_per_epoch, nenv*local_steps_per_epoch))

        for epoch in range(epochs):
            # training network
            # random.shuffle(ls_train)
            # for i, idx in enumerate(range(0, len(ls_train), batch_size)):
            #     # Perform batch update!
            #     # batch shape: [N, 224, 224]
            #     seg_pred_batch = seg_pred[ls_train[idx:idx+batch_size]]
            #     seg_gt_batch = seg_gt[ls_train[idx:idx+batch_size]]
            #     anchor_batch = anchor[ls_train[idx:idx+batch_size]].to(device)
            #     pos_batch = pos[ls_train[idx:idx+batch_size]].to(device)
            #     neg_batch = neg[ls_train[idx:idx+batch_size]].to(device)

            #     inputs = torch.cat((anchor_batch, pos_batch, neg_batch[0]), dim=0)
            #     inputs = inputs.unsqueeze(1)

            #     optimizer.zero_grad()
            #     outs = model(inputs)

            #     # anchor_feas, pos_feas, neg_feas = outs.chunk(3, dim=0)
            #     anchor_feas = outs[0:1]
            #     pos_feas = outs[1:2]
            #     neg_feas = outs[2:]

            #     anchor_flatten = anchor_feas.view(1,-1)
            #     pos_flatten = pos_feas.view(1,-1)
            #     neg_flatten = neg_feas.view(rot_num,-1)

            #     pos_dis = F.pairwise_distance(anchor_flatten, pos_flatten)
            #     neg_dis = F.pairwise_distance(anchor_flatten.tile((rot_num,1)), neg_flatten)
            #     neg_dis_clamp = torch.clamp(1-neg_dis, min=0).mean()
            #     loss = pos_dis + neg_dis_clamp

            #     loss.backward()
            #     optimizer.step()

            #     contrasAcc = 0 if neg_dis.le(pos_dis).any() else 1

            #     acc, acc_cls, mean_iu, fwavacc = label_accuracy_score(
            #         seg_gt_batch.numpy(), seg_pred_batch.numpy(), 3)

            #     logger.store(trainLoss=loss.item())
            #     logger.store(trainMeanIU=mean_iu.item())
            #     logger.store(trainContrasAcc=contrasAcc)

            # evaluating network
            with torch.no_grad():
                for idx in ls_test:
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

                    logger.store(evalLoss=loss.item())
                    logger.store(evalContrasAcc=contrasAcc)


            # Log info about epoch
            logger.log_tabular('Iterates', ite)
            logger.log_tabular('Epoch', epoch)
            # logger.log_tabular('trainLoss', average_only=True)
            # logger.log_tabular('trainContrasAcc', average_only=True)
            # logger.log_tabular('trainMeanIU', average_only=True)
            logger.log_tabular('evalLoss', average_only=True)
            logger.log_tabular('evalContrasAcc', average_only=True)
            logger.log_tabular('TotalEnvInteracts', (epoch+1)*local_steps_per_epoch*nenv)
            logger.log_tabular('Time', time.time()-start_time)
            logger.dump_tabular()

            # Save model
            if (epoch % save_freq == 0) or (epoch == epochs-1):
                logger.save_state({}, None)
                print('model saved !')



if __name__ == '__main__':
    pass
