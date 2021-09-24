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
 
 
 
class Buffer(object):
    def __init__(self, buffer_size, rot_num=6):
        super(Buffer, self).__init__()
        self.seg_pred_buf = np.zeros(shape=(buffer_size, 200, 250), dtype=np.int32)
        self.seg_gt_buf = np.zeros(shape=(buffer_size, 200, 250), dtype=np.int32)
        self.anchor_buf = np.zeros(shape=(buffer_size, 200, 250), dtype=np.float32)
        self.pos_buf = np.zeros(shape=(buffer_size, 200, 250), dtype=np.float32)
        self.neg_buf = np.zeros(shape=(buffer_size, rot_num, 200, 250), dtype=np.float32)
        self.dtheta_buf = np.zeros(shape=(buffer_size), dtype=np.float32)
        self.theta_pos_buf = np.zeros(shape=(buffer_size), dtype=np.float32)
        self.theta_neg_buf = np.zeros(shape=(buffer_size, rot_num), dtype=np.float32)
        self.ptr = 0

    def store(self, seg_pred, seg_gt, anchor, pos, neg, dtheta, theta_pos, theta_neg):
        self.seg_pred_buf[self.ptr] = seg_pred
        self.seg_gt_buf[self.ptr] = seg_gt
        self.anchor_buf[self.ptr] = anchor
        self.pos_buf[self.ptr] = pos
        self.neg_buf[self.ptr] = neg
        self.dtheta_buf[self.ptr] = dtheta
        self.theta_pos_buf[self.ptr] = theta_pos
        self.theta_neg_buf[self.ptr] = theta_neg
        self.ptr += 1

    def get(self):
        self.ptr = 0
        data = dict(seg_pred=self.seg_pred_buf, 
                    seg_gt=self.seg_gt_buf,
                    anchor=self.anchor_buf,
                    pos=self.pos_buf,
                    neg=self.neg_buf,
                    dtheta=self.dtheta_buf,
                    theta_pos=self.theta_pos_buf,
                    theta_neg=self.theta_neg_buf)
        return data


def transform(o):
        '''
        transform rgb image to anchor, positive and negative sample pair.
        '''
        img = o['img']
        gt_mask = o['gt']
        dtheta = o['dtheta']
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
        
        # for i in range(1):
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

        return seg_pred, peg_mask, base_mask_pos, base_mask_neg, dtheta, theta_pos, theta_neg
 
 

def test_pose(venv, nenv, seed=0, local_steps_per_epoch=1000, 
    device='cpu', model_path='', rot_num=10):
    # Random seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Load model
    model = torch.load(model_path, map_location=device)

    # Set up experience buffer
    buf_list = [Buffer(local_steps_per_epoch, rot_num) for _ in range(nenv)]


    # Main loop: collect experience in env and update/log each epoch
    o = venv.reset()
    # collecting data
    for t in range(local_steps_per_epoch):
        seg_pred, peg_mask, base_mask_pos, base_mask_neg, dtheta, theta_pos, theta_neg = transform(o)
        [buf_list[i].store(
                seg_pred[i], 
                o['gt'][i],
                peg_mask[i],
                base_mask_pos[i],
                base_mask_neg[i],
                dtheta[i],
                theta_pos[i],
                theta_neg[i]) for i in range(nenv)]

        # fake action
        o, _, _, _ = venv.step([0]*nenv)

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
    dtheta = torch.cat([torch.tensor(
        buf_list[i].get()['dtheta']) for i in range(nenv)])
    theta_pos = torch.cat([torch.tensor(
        buf_list[i].get()['theta_pos']) for i in range(nenv)])
    theta_neg = torch.cat([torch.tensor(
        buf_list[i].get()['theta_neg']) for i in range(nenv)])


    acc = 0
    # evaluating network
    with torch.no_grad():
        for idx in range(anchor.size(0)):
            gt_eval = seg_gt[idx]
            anchor_eval = anchor[idx:idx+1].to(device)
            pos_eval = pos[idx:idx+1].to(device)
            neg_eval = neg[idx].to(device)
            dtheta_eval = dtheta[idx]
            theta_pos_eval = theta_pos[idx]
            theta_neg_eval = theta_neg[idx]

            inputs = torch.cat((anchor_eval, pos_eval, neg_eval), dim=0)
            outs = model(inputs.unsqueeze(1))
            vis_outs = outs
            anchor_flatten = outs[0].view(1,-1)
            pos_flatten = outs[1].view(1,-1)
            neg_flatten = outs[2:].view(rot_num,-1)

            pos_dis = F.pairwise_distance(anchor_flatten, pos_flatten)
            neg_dis = F.pairwise_distance(anchor_flatten.repeat(rot_num,1), neg_flatten)
            neg_dis_clamp = torch.clamp(1-neg_dis, min=0).mean()
            loss = pos_dis + neg_dis_clamp

            contrasAcc = 0 if neg_dis.le(pos_dis).any() else 1

            acc += contrasAcc

            # print(idx, contrasAcc, loss)
            # print('neg_dis', neg_dis)
            # print('argmin', torch.argmin(neg_dis))
            # print(len(neg_dis), neg_flatten.shape)
            # exit(0)


            # plt.subplot(3,4,1)
            # img2 = vis_outs[1].detach().cpu().permute(1,2,0)-vis_outs[0].detach().cpu().permute(1,2,0)
            # img2 = (img2-torch.min(img2))/(torch.max(img2)-torch.min(img2))
            # plt.imshow(img2)
            # plt.subplot(3,4,2)
            # img3 = vis_outs[2].detach().cpu().permute(1,2,0)-vis_outs[1].detach().cpu().permute(1,2,0)
            # img3 = (img3-torch.min(img3))/(torch.max(img3)-torch.min(img3))
            # plt.imshow(img3)
            # plt.subplot(3,4,3)
            # img = vis_outs[3].detach().cpu().permute(1,2,0)-vis_outs[1].detach().cpu().permute(1,2,0)
            # img = (img-torch.min(img))/(torch.max(img)-torch.min(img))
            # plt.imshow(img)
            # plt.subplot(3,4,4)
            # img = vis_outs[4].detach().cpu().permute(1,2,0)-vis_outs[1].detach().cpu().permute(1,2,0)
            # img = (img-torch.min(img))/(torch.max(img)-torch.min(img))
            # plt.imshow(img)
            # plt.subplot(3,4,5)
            # img = vis_outs[5].detach().cpu().permute(1,2,0)-vis_outs[1].detach().cpu().permute(1,2,0)
            # img = (img-torch.min(img))/(torch.max(img)-torch.min(img))
            # plt.imshow(img)
            # plt.subplot(3,4,6)
            # img = vis_outs[6].detach().cpu().permute(1,2,0)-vis_outs[1].detach().cpu().permute(1,2,0)
            # img = (img-torch.min(img))/(torch.max(img)-torch.min(img))
            # plt.imshow(img)
            # plt.subplot(3,4,7)
            # img = vis_outs[7].detach().cpu().permute(1,2,0)-vis_outs[1].detach().cpu().permute(1,2,0)
            # img = (img-torch.min(img))/(torch.max(img)-torch.min(img))
            # plt.imshow(img)
            # plt.subplot(3,4,8)
            # img = vis_outs[8].detach().cpu().permute(1,2,0)-vis_outs[1].detach().cpu().permute(1,2,0)
            # img = (img-torch.min(img))/(torch.max(img)-torch.min(img))
            # plt.imshow(img)
            # plt.subplot(3,4,9)
            # img = vis_outs[9].detach().cpu().permute(1,2,0)-vis_outs[1].detach().cpu().permute(1,2,0)
            # img = (img-torch.min(img))/(torch.max(img)-torch.min(img))
            # plt.imshow(img)
            # plt.subplot(3,4,10)
            # img = vis_outs[10].detach().cpu().permute(1,2,0)-vis_outs[1].detach().cpu().permute(1,2,0)
            # img = (img-torch.min(img))/(torch.max(img)-torch.min(img))
            # plt.imshow(img)
            # plt.subplot(3,4,11)
            # img = vis_outs[11].detach().cpu().permute(1,2,0)-vis_outs[1].detach().cpu().permute(1,2,0)
            # img = (img-torch.min(img))/(torch.max(img)-torch.min(img))
            # plt.imshow(img)
            # plt.subplot(3,4,12)
            # plt.imshow(gt_eval)
            # plt.show()

            # ax1 = plt.subplot(3,4,1)
            # ax1.set_title(str(dtheta_eval)+'/'+str(theta_pos_eval) + str(pos_dis.item()))
            # plt.imshow(cv2.addWeighted(np.array(pos_eval[0].detach().cpu()), 1, np.array(anchor_eval[0].detach().cpu()), 2, 0))
            # ax2 = plt.subplot(3,4,2)
            # ax2.set_title(str(theta_neg_eval[0]) + str(neg_dis[0].item()))
            # plt.imshow(cv2.addWeighted(np.array(neg_eval[0].detach().cpu()), 1, np.array(anchor_eval[0].detach().cpu()), 2, 0))
            # ax3 = plt.subplot(3,4,3)
            # ax3.set_title(str(theta_neg_eval[1]) + str(neg_dis[1].item()))
            # plt.imshow(cv2.addWeighted(np.array(neg_eval[1].detach().cpu()), 1, np.array(anchor_eval[0].detach().cpu()), 2, 0))
            # ax4 = plt.subplot(3,4,4)
            # ax4.set_title(str(theta_neg_eval[2]) + str(neg_dis[2].item()))
            # plt.imshow(cv2.addWeighted(np.array(neg_eval[2].detach().cpu()), 1, np.array(anchor_eval[0].detach().cpu()), 2, 0))
            # ax5 = plt.subplot(3,4,5)
            # ax5.set_title(str(theta_neg_eval[3]) + str(neg_dis[3].item()))
            # plt.imshow(cv2.addWeighted(np.array(neg_eval[3].detach().cpu()), 1, np.array(anchor_eval[0].detach().cpu()), 2, 0))
            # ax6 = plt.subplot(3,4,6)
            # ax6.set_title(str(theta_neg_eval[4]) + str(neg_dis[4].item()))
            # plt.imshow(cv2.addWeighted(np.array(neg_eval[4].detach().cpu()), 1, np.array(anchor_eval[0].detach().cpu()), 2, 0))
            # ax7 = plt.subplot(3,4,7)
            # ax7.set_title(str(theta_neg_eval[5]) + str(neg_dis[5].item()))
            # plt.imshow(cv2.addWeighted(np.array(neg_eval[5].detach().cpu()), 1, np.array(anchor_eval[0].detach().cpu()), 2, 0))
            # ax8 = plt.subplot(3,4,8)
            # ax8.set_title(str(theta_neg_eval[6]) + str(neg_dis[6].item()))
            # plt.imshow(cv2.addWeighted(np.array(neg_eval[6].detach().cpu()), 1, np.array(anchor_eval[0].detach().cpu()), 2, 0))
            # ax9 = plt.subplot(3,4,9)
            # ax9.set_title(str(theta_neg_eval[7]) + str(neg_dis[7].item()))
            # plt.imshow(cv2.addWeighted(np.array(neg_eval[7].detach().cpu()), 1, np.array(anchor_eval[0].detach().cpu()), 2, 0))
            # ax10 = plt.subplot(3,4,10)
            # ax10.set_title(str(theta_neg_eval[8]) + str(neg_dis[8].item()))
            # plt.imshow(cv2.addWeighted(np.array(neg_eval[8].detach().cpu()), 1, np.array(anchor_eval[0].detach().cpu()), 2, 0))
            # ax10 = plt.subplot(3,4,11)
            # ax10.set_title(str(theta_neg_eval[9]) + str(neg_dis[9].item()))
            # plt.imshow(cv2.addWeighted(np.array(neg_eval[9].detach().cpu()), 1, np.array(anchor_eval[0].detach().cpu()), 2, 0))
            # plt.show()

    print('contrasAcc', acc/anchor.size(0))


 

if __name__ == '__main__':
    pass
