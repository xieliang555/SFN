import numpy as np
np.set_printoptions(threshold=np.inf)
import torch
import torch.nn.functional as F
from torch.optim import Adam
import gym
import time
import algos.pytorch.a2c_9.core as core
from utils.logx import EpochLogger
import matplotlib.pyplot as plt
from PIL import Image
import cv2
  
 
class PGBuffer(object):
    """
    A buffer for storing trajectories experienced by a PG agent interacting 
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda) 
    for calculating the advantages of state-action pairs.
    """
    def __init__(self, buffer_size, obs_dim, neg_num, gamma=0.99, lam=0.95):
        super(PGBuffer, self).__init__()
        self.peg_mask_buf = np.zeros(shape=(buffer_size, *obs_dim), dtype=np.float32)
        self.hole_mask_buf = np.zeros(shape=(buffer_size, *obs_dim), dtype=np.float32)
        self.hole_mask_pos_buf = np.zeros(shape=(buffer_size, *obs_dim), dtype=np.float32)
        self.hole_mask_neg_buf = np.zeros(shape=(buffer_size, neg_num, *obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(shape=(buffer_size), dtype=np.float32)
        self.rew_buf = np.zeros(shape=(buffer_size), dtype=np.float32)
        self.val_buf = np.zeros(shape=(buffer_size), dtype=np.float32)
        self.ret_buf = np.zeros(shape=(buffer_size), dtype=np.float32)
        self.adv_buf = np.zeros(shape=(buffer_size), dtype=np.float32)
        self.ptr, self.path_start_idx = 0, 0
        self.gamma, self.lam = gamma, lam


    def store(self, peg_mask, hole_mask, hole_mask_pos, hole_mask_neg, act, rew=0, val=0):
        self.peg_mask_buf[self.ptr] = peg_mask
        self.hole_mask_buf[self.ptr] = hole_mask
        self.hole_mask_pos_buf[self.ptr] = hole_mask_pos
        self.hole_mask_neg_buf[self.ptr] = hole_mask_neg
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.ptr += 1


    def finish_path(self, last_val):
        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)
        deltas = rews[:-1]+self.gamma*vals[1:]-vals[:-1]
        # !!!!!!! 
        # replace discount_cumsum
        self.ret_buf[path_slice] = core.discount_cumsum(rews, self.gamma)[:-1]
        self.adv_buf[path_slice] = core.discount_cumsum(deltas, self.gamma*self.lam)

        self.path_start_idx = self.ptr
        

    def get(self):
        self.ptr, self.path_start_idx = 0, 0
        data = dict(peg_mask=self.peg_mask_buf, hole_mask=self.hole_mask_buf,
                    hole_mask_pos=self.hole_mask_pos_buf, hole_mask_neg=self.hole_mask_neg_buf,
                    act=self.act_buf, ret=self.ret_buf, adv=self.adv_buf)
        return data




def a2c(venv, nenv, ac_kwargs=dict(), seed=0, 
        local_steps_per_epoch=1000, epochs=50, gamma=0.99, 
        train_v_iters=80, lam=0.97, logger_kwargs=dict(), neg_num=6,
        save_freq=10, device='cpu', resume=False, model_path=''):
    """
    Vanilla Policy Gradient 

    (with GAE-Lambda for advantage estimation)

    Args:
        env_fn : A function which creates a copy of the environment.
            The environment must satisfy the OpenAI Gym API.

        actor_critic: The constructor method for a PyTorch Module with a 
            ``step`` method, an ``act`` method, a ``pi`` module, and a ``v`` 
            module. The ``step`` method should accept a batch of observations 
            and return:

            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``a``        (batch, act_dim)  | Numpy array of actions for each 
                                           | observation.
            ``v``        (batch,)          | Numpy array of value estimates
                                           | for the provided observations.
            ``logp_a``   (batch,)          | Numpy array of log probs for the
                                           | actions in ``a``.
            ===========  ================  ======================================

            The ``act`` method behaves the same as ``step`` but only returns ``a``.

            The ``pi`` module's forward call should accept a batch of 
            observations and optionally a batch of actions, and return:

            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``pi``       N/A               | Torch Distribution object, containing
                                           | a batch of distributions describing
                                           | the policy for the provided observations.
            ``logp_a``   (batch,)          | Optional (only returned if batch of
                                           | actions is given). Tensor containing 
                                           | the log probability, according to 
                                           | the policy, of the provided actions.
                                           | If actions not given, will contain
                                           | ``None``.
            ===========  ================  ======================================

            The ``v`` module's forward call should accept a batch of observations
            and return:

            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``v``        (batch,)          | Tensor containing the value estimates
                                           | for the provided observations. (Critical: 
                                           | make sure to flatten this!)
            ===========  ================  ======================================

        ac_kwargs (dict): Any kwargs appropriate for the ActorCritic object 
            you provided to VPG.

        seed (int): Seed for random number generators.

        steps_per_epoch (int): Number of steps of interaction (state-action pairs) 
            for the agent and the environment in each epoch.

        epochs (int): Number of epochs of interaction (equivalent to
            number of policy updates) to perform.

        gamma (float): Discount factor. (Always between 0 and 1.)

        pi_lr (float): Learning rate for policy optimizer.

        vf_lr (float): Learning rate for value function optimizer.

        train_v_iters (int): Number of gradient descent steps to take on 
            value function per epoch.

        lam (float): Lambda for GAE-Lambda. (Always between 0 and 1,
            close to 1.)

        max_ep_len (int): Maximum length of trajectory / episode / rollout.

        logger_kwargs (dict): Keyword args for EpochLogger.

        save_freq (int): How often (in terms of gap between epochs) to save
            the current policy and value function.

    """

    # Set up logger and save configuration
    logger = EpochLogger(**logger_kwargs)
    # logger.save_config(locals())

    # Random seed
    seed = 0
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Instantiate environment
    obs_dim = venv.observation_space.shape[1:]

    # Create actor-critic module
    if resume:
        ac = torch.load(model_path, map_location=device)
    else:
        ac = core.A2C(venv.action_space).to(device)

    # Set up experience buffer
    buf_list = [PGBuffer(local_steps_per_epoch, obs_dim, neg_num, gamma, lam) for _ in range(nenv)]


    def transform(o):
        gt_mask = o['gt']
        dx = o['dx']
        dy = o['dy']
        dtheta = o['dtheta']
        peg_mask = np.uint8(gt_mask==1)
        hole_mask = np.uint8(gt_mask==2)

        # transform 
        hole_mask_pos = [Image.fromarray(hole_mask[i]).rotate(
            dtheta[i], expand=False, fillcolor=0, translate=(-dx[i],dy[i])) for i in range(nenv)]
        hole_mask_neg = []
        for n in range(nenv):
            hole_mask_neg_ls = []
            for i in range(neg_num):
                # 15 -> 0.1
                # !!!!!!!!!!!
                mask = Image.fromarray(hole_mask[n]).rotate(
                    np.random.uniform(-0.1,0.1), expand=False, fillcolor=0, 
                    translate=tuple(np.random.uniform(-30,30,2)))
                hole_mask_neg_ls.append(np.array(mask))
            hole_mask_neg.append(hole_mask_neg_ls)

        #     plt.subplot(2,4,1)
        #     plt.imshow(cv2.addWeighted(np.array(hole_mask_pos[n])*2,1,peg_mask[n],1,0))
        #     plt.subplot(2,4,2)
        #     plt.imshow(cv2.addWeighted(np.array(hole_mask_neg[n][0])*2,1,peg_mask[n],1,0))
        #     plt.subplot(2,4,3)
        #     plt.imshow(cv2.addWeighted(np.array(hole_mask_neg[n][1])*2,1,peg_mask[n],1,0))
        #     plt.subplot(2,4,4)
        #     plt.imshow(cv2.addWeighted(np.array(hole_mask_neg[n][2])*2,1,peg_mask[n],1,0))
        #     plt.subplot(2,4,5)
        #     plt.imshow(cv2.addWeighted(np.array(hole_mask_neg[n][3])*2,1,peg_mask[n],1,0))
        #     plt.subplot(2,4,6)
        #     plt.imshow(cv2.addWeighted(np.array(hole_mask_neg[n][4])*2,1,peg_mask[n],1,0))
        #     plt.subplot(2,4,7)
        #     plt.imshow(cv2.addWeighted(np.array(hole_mask_neg[n][5])*2,1,peg_mask[n],1,0))
        #     plt.show()

        # exit(0)

        return hole_mask_pos, hole_mask_neg


    # Set up function for computing VPG policy loss
    def compute_loss_a2c(peg_mask, hole_mask, act, adv, ret):
        logp, v = ac(torch.stack((peg_mask, hole_mask), 0).unsqueeze(1), act)
        loss_pi = -(logp * adv).mean()
        loss_v = ((v - ret)**2).mean()
        loss_a2c = loss_v + loss_pi
        return loss_a2c

    def compute_contrastive_loss(peg_mask, hole_mask_pos, hole_mask_neg):
        logits = ac.feature_extractor(torch.cat((peg_mask.unsqueeze(0), 
            hole_mask_pos.unsqueeze(0),hole_mask_neg), 0).unsqueeze(1))
        anchor = logits[0:1]
        pos = logits[1:2]
        neg = logits[2:]

        pos_dis = F.pairwise_distance(anchor, pos)
        neg_dis = F.pairwise_distance(torch.tile(anchor, (neg_num,1)), neg)
        pos_dis_clamp = torch.clamp(pos_dis-0.1, min=0)
        neg_dis_clamp = torch.clamp(1-neg_dis, min=0).mean()
        loss_contrastive = pos_dis_clamp + neg_dis_clamp
        acc_contrastive = 0 if neg_dis.le(pos_dis).any() else 1
        return loss_contrastive, acc_contrastive

    # Set up optimizers for policy and value function
    a2c_optimizer = Adam(ac.parameters(), lr=1e-4)

    # Set up model saving
    logger.setup_pytorch_saver(ac)

    def update():
        peg_mask = torch.cat([torch.tensor(
            buf_list[i].get()['peg_mask']) for i in range(nenv)]).to(device)
        hole_mask = torch.cat([torch.tensor(
            buf_list[i].get()['hole_mask']) for i in range(nenv)]).to(device)
        hole_mask_pos = torch.cat([torch.tensor(
            buf_list[i].get()['hole_mask_pos']) for i in range(nenv)]).to(device)
        hole_mask_neg = torch.cat([torch.tensor(
            buf_list[i].get()['hole_mask_neg']) for i in range(nenv)]).to(device)
        act = torch.cat([torch.tensor(
            buf_list[i].get()['act'], dtype=torch.long) for i in range(nenv)]).to(device)
        adv = torch.cat([torch.tensor(
            buf_list[i].get()['adv']) for i in range(nenv)]).to(device)
        adv = (adv - torch.mean(adv)) / torch.std(adv)
        ret = torch.cat([torch.tensor(
            buf_list[i].get()['ret']) for i in range(nenv)]).to(device)


        # iterates the epoch
        for i in range(len(peg_mask)):
            # visualize
            print(i, 'act', act, 'adv', adv, 'ret', ret)
            plt.subplot(3,4,1)
            plt.imshow(peg_mask[i].detach().cpu())
            plt.subplot(3,4,2)
            plt.imshow(hole_mask[i].detach().cpu())
            plt.subplot(3,4,3)
            plt.imshow(cv2.addWeighted(np.array(hole_mask_pos[i].detach().cpu())*2,1,np.array(peg_mask[i].detach().cpu()),1,0))
            plt.subplot(3,4,4)
            plt.imshow(cv2.addWeighted(np.array(hole_mask_neg[i][0].detach().cpu())*2,1,np.array(peg_mask[i].detach().cpu()),1,0))
            plt.subplot(3,4,5)
            plt.imshow(cv2.addWeighted(np.array(hole_mask_neg[i][1].detach().cpu())*2,1,np.array(peg_mask[i].detach().cpu()),1,0))
            plt.subplot(3,4,6)
            plt.imshow(cv2.addWeighted(np.array(hole_mask_neg[i][2].detach().cpu())*2,1,np.array(peg_mask[i].detach().cpu()),1,0))
            plt.subplot(3,4,7)
            plt.imshow(cv2.addWeighted(np.array(hole_mask_neg[i][3].detach().cpu())*2,1,np.array(peg_mask[i].detach().cpu()),1,0))
            plt.subplot(3,4,8)
            plt.imshow(cv2.addWeighted(np.array(hole_mask_neg[i][4].detach().cpu())*2,1,np.array(peg_mask[i].detach().cpu()),1,0))
            plt.subplot(3,4,9)
            plt.imshow(cv2.addWeighted(np.array(hole_mask_neg[i][5].detach().cpu())*2,1,np.array(peg_mask[i].detach().cpu()),1,0))
            plt.show()



            # Train policy with a single step of gradient descent
            a2c_optimizer.zero_grad()
            loss_a2c = compute_loss_a2c(peg_mask[i], hole_mask[i], act[i], adv[i], ret[i])
            loss_contrastive, acc_contrastive = compute_contrastive_loss(
                peg_mask[i], hole_mask_pos[i], hole_mask_neg[i])
            loss = loss_a2c + loss_contrastive
            loss.backward()
            a2c_optimizer.step()

            # Log changes from update
            logger.store(LossA2C=loss_a2c.item())
            logger.store(LossContrastive=loss_contrastive.item())
            logger.store(AccConstractive=acc_contrastive)
            logger.store(Loss=loss.item())

    # Prepare for interaction with environment
    start_time = time.time()

    # Main loop: collect experience in env and update/log each epoch
    for epoch in range(epochs):
        o, ep_ret, ep_len = venv.reset(), np.zeros(nenv), np.zeros(nenv)

        for t in range(local_steps_per_epoch):
            peg_mask = torch.tensor(o['gt']==1, dtype=torch.float32)
            hole_mask = torch.tensor(o['gt']==2, dtype=torch.float32)
            inputs = torch.cat((peg_mask, hole_mask), 0).unsqueeze(1).to(device)
            a, v = ac.step(inputs)

            next_o, r, d, info = venv.step(a.detach().cpu())

            ep_ret += r
            ep_len += 1

            hole_mask_pos, hole_mask_neg = transform(o)

            # save and log
            [buf_list[i].store(
                peg_mask[i],hole_mask[i],hole_mask_pos[i],
                hole_mask_neg[i],a[i],r[i],v[i]) for i in range(nenv)]
            logger.store(VVals=torch.mean(v))
            
            # Update obs (critical!)
            o = next_o
            
            # finish path if done
            if t == local_steps_per_epoch-1:
                _, last_val = ac.step(torch.cat((
                    torch.tensor(o['gt']==1, dtype=torch.float32),
                    torch.tensor(o['gt']==2, dtype=torch.float32)), 0).unsqueeze(1).to(device))
                last_val = last_val.cpu()
                [buf_list[i].finish_path(last_val[i]) for i in range(nenv)]
            elif d.any():
                last_val = []
                for i in info:
                    if i:
                        inputs = torch.stack((
                            torch.tensor(i['ob_next']['gt']==1, dtype=torch.float32),
                            torch.tensor(i['ob_next']['gt']==2, dtype=torch.float32)), 0).unsqueeze(1).to(device)
                        _, val = ac.step(inputs)
                        last_val.append(val.item())
                    else:
                        last_val.append(0)

                idx = np.where(d)[0]
                idx_done = [i for i in idx if r[i]>0]
                idx_timeout = [i for i in idx if r[i]==0]
                [buf_list[i].finish_path(0) for i in idx_done]
                [buf_list[i].finish_path(last_val[i]) for i in idx_timeout]

                [logger.store(EpRet=ep_ret[i], EpLen=ep_len[i]) for i in idx]
                ep_len[idx] = 0
                ep_ret[idx] = 0
                

        # Save model
        if (epoch % save_freq == 0) or (epoch == epochs-1):
            logger.save_state({}, None)
            print('model saved')

        # Perform VPG update!
        update()

        # Log info about epoch
        logger.log_tabular('Epoch', epoch)
        logger.log_tabular('EpRet', with_min_and_max=True)
        logger.log_tabular('EpLen', average_only=True)
        logger.log_tabular('VVals', with_min_and_max=True)
        logger.log_tabular('LossA2C', average_only=True)
        logger.log_tabular('LossContrastive', average_only=True)
        logger.log_tabular('Loss', average_only=True)
        logger.log_tabular('AccConstractive', average_only=True)
        logger.log_tabular('TotalEnvInteracts', (epoch+1)*local_steps_per_epoch*nenv)
        logger.log_tabular('Time', time.time()-start_time)
        logger.dump_tabular()


if __name__ == '__main__':
    pass




