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
from dataloader import UR5Dataset
from torch.utils.data import DataLoader
 
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



def train_seg(seed=0, epochs=50, logger_kwargs=dict(), save_freq=10, 
    device='cpu', resume=False, model_path=''):

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
        model = UNet(3,3)
        model =model.to(device)
    optimizer = Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    trainSet = UR5Dataset(root='/home/xieliang/data/ur5', 
        annotation_path='/home/xieliang/data/ur5/annotation_train.csv')
    testSet = UR5Dataset(root='/home/xieliang/data/ur5', 
        annotation_path='/home/xieliang/data/ur5/annotation_test.csv')
    trainLoader = DataLoader(trainSet, batch_size=1, shuffle=True, 
        num_workers=0, pin_memory=True)
    testLoader = DataLoader(testSet, batch_size=10, shuffle=False,
        num_workers=0, pin_memory=True)

    # Set up model saving
    logger.setup_pytorch_saver(model)
    start_time = time.time()

    # training network
    for epoch in range(epochs):
        for batch_idx, batch in enumerate(trainLoader):
            img = batch['img'].float().to(device)
            seg_gt = (batch['seg_gt']/100.).long().to(device)



            # print(img.shape)
            # print(seg_gt.shape)
            # plt.subplot(1,2,1)
            # plt.imshow(img[0].cpu()/255.)
            # plt.subplot(1,2,2)
            # plt.imshow(seg_gt[0].cpu())
            # plt.show()
            # continue

            preds = model(img.permute(0,3,1,2))

            # preds->preds_flatten: [n,c,h,w]->[n,h,w,c]->[n*h*w,c]
            n,c,h,w = preds.size()
            preds_flatten = preds.permute(0,2,3,1).contiguous().view(-1,c)
            gts_flatten = seg_gt.view(-1)


            optimizer.zero_grad()
            loss = criterion(preds_flatten, gts_flatten)
            loss.backward()
            optimizer.step()

            preds_batch = preds.max(1)[1]
            acc, acc_cls, mean_iu, fwavacc = label_accuracy_score(
                seg_gt.cpu().numpy(), preds_batch.cpu().numpy(), 3)
            
            logger.store(trainLoss=loss.item())
            logger.store(trainAcc=acc.item())
            logger.store(trainAccCls=acc_cls.item())
            logger.store(trainMeanIU=mean_iu.item())
            logger.store(trainFWAVAcc=fwavacc.item())

        # evaluating
        with torch.no_grad():
            for batch_idx, batch in enumerate(testLoader):
                img = batch['img'].float().to(device)
                seg_gt = (batch['seg_gt']/100.).long().to(device)

                
                preds = model(img.permute(0,3,1,2))

                n,c,h,w = preds.size()
                preds_flatten = preds.permute(0,2,3,1).contiguous().view(-1,c)
                gts_flatten = seg_gt.view(-1)

                loss = criterion(preds_flatten, gts_flatten)

                preds_batch = preds.max(1)[1].cpu().numpy()
                gt_batch = seg_gt.cpu().numpy()
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
        logger.log_tabular('Epoch', epoch)
        logger.log_tabular('trainLoss', average_only=True)
        logger.log_tabular('trainMeanIU', average_only=True)
        logger.log_tabular('evalLoss', average_only=True)
        logger.log_tabular('evalMeanIU', average_only=True)
        logger.log_tabular('Time', time.time()-start_time)
        # trainMeanIU = logger.log_current_row['trainMeanIU']
        logger.dump_tabular()


        if (epoch % save_freq == 0) or (epoch == epochs-1):
            logger.save_state({}, None)
            print('final model saved !')




if __name__ == '__main__':
    pass
