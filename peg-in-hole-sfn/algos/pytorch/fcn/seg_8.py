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

 

def train_seg(venv, nenv, seed=0, local_steps_per_epoch=1000, 
    epochs=50, logger_kwargs=dict(), save_freq=10, device='cpu', 
    resume=False, model_path='', iterates=100, batch_size=10):

    # Set up logger and save configuration
    logger = EpochLogger(**logger_kwargs)
    # logger.save_config(locals())

    # Random seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Load model
    if resume:
        model = torch.load(model_path).to(device)
    else:
        model = UNet(3,3).to(device)

    # Set up experience buffer
    buf_list = [Buffer(local_steps_per_epoch) for _ in range(nenv)]

    optimizer = Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()


    # Set up model saving
    logger.setup_pytorch_saver(model)

    start_time = time.time()

    # Main loop: collect experience in env and update/log each epoch
    for ite in range(iterates):
        o = venv.reset()

        # collecting data
        for t in range(local_steps_per_epoch):
            [buf_list[i].store(
                o['img'][i],
                o['gt'][i]) for i in range(nenv)]
            # fake action
            o, _, _, _ = venv.step([0]*nenv)


        obs_img = torch.cat([torch.tensor(
            buf_list[i].get()['obs_img']) for i in range(nenv)])
        gt = torch.cat([torch.tensor(
            buf_list[i].get()['gt'], dtype=torch.long) for i in range(nenv)])

        # split training and testing dataset
        ls_train = list(range((nenv-1)*local_steps_per_epoch))
        ls_test = list(range((nenv-1)*local_steps_per_epoch,nenv*local_steps_per_epoch))


        # training network
        for epoch in range(epochs):
            random.shuffle(ls_train)
            for i, idx in enumerate(range(0, len(ls_train), batch_size)):
                # Perform batch update!
                img_batch = obs_img[ls_train[idx:idx+batch_size]].to(device)
                gt_batch = gt[ls_train[idx:idx+batch_size]].to(device)

                # print(img_batch.shape)
                # print(gt_batch.shape)
                # plt.subplot(1,2,1)
                # plt.imshow(img_batch[0].cpu().permute((1,2,0))/255)
                # plt.subplot(1,2,2)
                # plt.imshow(gt_batch[0].cpu())
                # plt.show()
                # continue

                preds = model(img_batch)

                # preds->preds_flatten: [n,c,h,w]->[n,h,w,c]->[n*h*w,c]
                n,c,h,w = preds.size()
                preds_flatten = preds.permute(0,2,3,1).contiguous().view(-1,c)
                gts_flatten = gt_batch.view(-1)

                optimizer.zero_grad()
                loss = criterion(preds_flatten, gts_flatten)
                loss.backward()
                optimizer.step()

                preds_batch = preds.max(1)[1]
                acc, acc_cls, mean_iu, fwavacc = label_accuracy_score(
                    gt_batch.cpu().numpy(), preds_batch.cpu().numpy(), 3)
                
                logger.store(trainLoss=loss.item())
                logger.store(trainAcc=acc.item())
                logger.store(trainAccCls=acc_cls.item())
                logger.store(trainMeanIU=mean_iu.item())
                logger.store(trainFWAVAcc=fwavacc.item())

            # evaluating
            with torch.no_grad():
                for i, idx in enumerate(range(0, len(ls_test), batch_size)):
                    img_eval = obs_img[ls_test[idx:idx+batch_size]].to(device)
                    gt_eval = gt[ls_test[idx:idx+batch_size]].to(device)
                    
                    preds = model(img_eval)

                    n,c,h,w = preds.size()
                    preds_flatten = preds.permute(0,2,3,1).contiguous().view(-1,c)
                    gts_flatten = gt_eval.view(-1)

                    loss = criterion(preds_flatten, gts_flatten)

                    preds_batch = preds.max(1)[1].cpu().numpy()
                    gt_batch = gt_eval.cpu().numpy()
                    acc, acc_cls, mean_iu, fwavacc = label_accuracy_score(
                        gt_batch, preds_batch, 3)

                    logger.store(evalLoss=loss.item())
                    logger.store(evalMeanIU=mean_iu.item())


                    # plt.subplot(1,3,1)
                    # plt.imshow(img_eval[0].cpu().detach().permute(1,2,0)/255)
                    # plt.subplot(1,3,2)
                    # plt.imshow(preds_batch[0])
                    # plt.subplot(1,3,3)
                    # plt.imshow(gt_batch[0])
                    # plt.show()


            # Log info about epoch
            logger.log_tabular('Iterates', ite)
            logger.log_tabular('Epoch', epoch)
            logger.log_tabular('trainLoss', average_only=True)
            logger.log_tabular('trainMeanIU', average_only=True)
            logger.log_tabular('evalLoss', average_only=True)
            logger.log_tabular('evalMeanIU', average_only=True)
            logger.log_tabular('TotalEnvInteracts', (epoch+1)*local_steps_per_epoch*nenv)
            logger.log_tabular('Time', time.time()-start_time)
            logger.dump_tabular()


            if (epoch % save_freq == 0) or (epoch == epochs-1):
                logger.save_state({}, None)
                print('final model saved !')



if __name__ == '__main__':
    pass
