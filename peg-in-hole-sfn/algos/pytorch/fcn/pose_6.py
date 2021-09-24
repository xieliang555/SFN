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
from torch.utils.tensorboard import SummaryWriter
 

 
class Buffer(object):
    def __init__(self, buffer_size):
        super(Buffer, self).__init__()
        self.img_buf = np.zeros(shape=(buffer_size,3,400,500), dtype=np.float32)
        self.dtheta_buf = np.zeros(shape=(buffer_size,), dtype=np.float32)
        self.ptr = 0

    def store(self, img, dtheta):
        self.img_buf[self.ptr] = img
        self.dtheta_buf[self.ptr] = dtheta
        self.ptr += 1

    def get(self):
        self.ptr = 0
        data = dict(img=self.img_buf,
                    dtheta=self.dtheta_buf)
        return data


def rot_img(x, theta, device):
    # Rotation by theta with autograd support
    # x: [N,C,H,W]
    theta = torch.tensor(theta)
    rot_mat = torch.tensor([
        [torch.cos(theta), -torch.sin(theta), 0],
        [torch.sin(theta), torch.cos(theta), 0]])
    rot_mat = rot_mat[None, ...].repeat(x.shape[0],1,1)
    grid = F.affine_grid(rot_mat, x.size(), align_corners=True).to(device)
    x = F.grid_sample(x, grid, align_corners=True)
    return x



def train_pose(venv, nenv, seed=0, local_steps_per_epoch=1000, 
    epochs=50, logger_kwargs=dict(), save_freq=10, device='cpu', 
    resume=False, model_path='', iterates=100, batch_size=10):

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
        model = UNet(3,128).to(device)

    # Set up experience buffer
    buf_list = [Buffer(local_steps_per_epoch) for _ in range(nenv)]
    optimizer = Adam(model.parameters(), lr=1e-4)
    writer = SummaryWriter()


    # Set up model saving
    logger.setup_pytorch_saver(model)

    start_time = time.time()

    # Main loop: collect experience in env and update/log each epoch
    for ite in range(iterates):
        o = venv.reset()
        # collecting data
        for t in range(local_steps_per_epoch):
            [buf_list[i].store(o['img'][i], o['dtheta'][i]) for i in range(nenv)]

            # fake action
            o, _, _, _ = venv.step([0]*nenv)

        img = torch.cat([torch.tensor(
            buf_list[i].get()['img']) for i in range(nenv)])
        dtheta = torch.cat([torch.tensor(
            buf_list[i].get()['dtheta']) for i in range(nenv)])


        # split training and testing dataset
        # !!!!!!!!!!!!!1
        # ls_train = list(range((nenv-1)*local_steps_per_epoch))
        # ls_test = list(range((nenv-1)*local_steps_per_epoch, nenv*local_steps_per_epoch))
        ls_train = list(range((nenv)*local_steps_per_epoch))

        for epoch in range(epochs):
            # training network
            random.shuffle(ls_train)
            for idx in ls_train:
                img_batch = img[idx:idx+1].to(device)
                dtheta_batch = dtheta[idx:idx+1]

                # print('dtheta', dtheta_batch)
                # img_batch_pos = rot_img(img_batch, dtheta_batch[0], device)
                # img_batch_neg = rot_img(img_batch, -dtheta_batch[0], device)
                # plt.subplot(1,3,1)
                # plt.imshow(img_batch[0].detach().cpu().permute(1,2,0)/255)
                # plt.subplot(1,3,2)
                # plt.imshow(img_batch_pos[0].detach().cpu().permute(1,2,0)/255)
                # plt.subplot(1,3,3)
                # plt.imshow(img_batch_neg[0].detach().cpu().permute(1,2,0)/255)
                # plt.show()
                # continue

                optimizer.zero_grad()
                outs = model(img_batch)

                peg_feas, base_feas = outs.chunk(2, dim=1)
                # !!!!!!!!!!!
                rot_list = np.array([-30,-20,-10,0,10,20,30])
                rot_num = len(rot_list)-1
                # rot_list = np.array([-30, -20, -15, -10, -9, -7, -5, -3, -1, 0, 1, 3, 5, 7, 9, 10, 15, 20, 30])
                # !!!!!!!
                pos_theta = rot_list[abs((-dtheta_batch) - rot_list).argmin()]
                neg_theta = [i for i in rot_list if i != pos_theta]
                pos_feas = rot_img(base_feas, pos_theta, device)
                neg_feas = torch.cat([rot_img(base_feas, i, device) for i in neg_theta],0)

                anchor = peg_feas.view(1,-1)
                pos = pos_feas.view(1,-1)
                neg = neg_feas.view(rot_num,-1)

                pos_dis = F.pairwise_distance(anchor, pos)
                neg_dis = F.pairwise_distance(anchor.tile((rot_num,1)), neg)
                neg_dis_clamp = torch.clamp(1-neg_dis, min=0).mean()
                loss = pos_dis + neg_dis_clamp

                loss.backward()
                optimizer.step()

                contrasAcc = 0 if neg_dis.le(pos_dis).any() else 1

                logger.store(trainLoss=loss.item())
                logger.store(trainContrasAcc=contrasAcc)

                writer.add_scalar('pos_dis', pos_dis, idx+epoch*epochs+ite*iterates)
                writer.add_scalar('neg_dis', neg_dis.mean(), idx+epoch*epochs+ite*iterates)
                writer.add_scalar('contrasAcc', contrasAcc, idx+epoch*epochs+ite*iterates)

            # # evaluating network
            # with torch.no_grad():
            #     for idx in ls_test:
            #         img_batch = img[idx:idx+1].to(device)
            #         dtheta_batch = dtheta[idx:idx+1]

            #         outs = model(img_batch)

            #         peg_feas, base_feas = outs.chunk(2, dim=1)
            #         rot_list = np.array([-30, -20, -15, -10, -9, -7, -5, -3, -1, 0, 1, 3, 5, 7, 9, 10, 15, 20, 30])
            # !!!!!!!!! -dtheta_batch ?
            #         pos_theta = rot_list[abs(dtheta_batch - rot_list).argmin()]
            #         neg_theta = [i for i in rot_list if i != pos_theta]
            #         pos_feas = rot_img(base_feas, pos_theta, device)
            #         neg_feas = torch.cat([rot_img(base_feas, i, device) for i in neg_theta],0)

            #         anchor = peg_feas.view(1,-1)
            #         pos = pos_feas.view(1,-1)
            #         neg = neg_feas.view(rot_num,-1)

            #         pos_dis = F.pairwise_distance(anchor, pos)
            #         neg_dis = F.pairwise_distance(anchor.tile((rot_num,1)), neg)
            #         neg_dis_clamp = torch.clamp(1-neg_dis, min=0).mean()
            #         loss = pos_dis + neg_dis_clamp

            #         contrasAcc = 0 if neg_dis.le(pos_dis).any() else 1

            #         logger.store(evalLoss=loss.item())
            #         logger.store(evalContrasAcc=contrasAcc)



            # Log info about epoch
            logger.log_tabular('Iterates', ite)
            logger.log_tabular('Epoch', epoch)
            logger.log_tabular('trainLoss', average_only=True)
            logger.log_tabular('trainContrasAcc', average_only=True)
            # logger.log_tabular('evalLoss', average_only=True)
            # logger.log_tabular('evalContrasAcc', average_only=True)
            logger.log_tabular('TotalEnvInteracts', (epoch+1)*local_steps_per_epoch*nenv)
            logger.log_tabular('Time', time.time()-start_time)
            logger.dump_tabular()

            # Save model
            if (epoch % save_freq == 0) or (epoch == epochs-1):
                logger.save_state({}, None)
                print('model saved !')



if __name__ == '__main__':
    pass
