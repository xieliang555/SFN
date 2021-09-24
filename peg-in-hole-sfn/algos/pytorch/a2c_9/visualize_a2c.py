import numpy as np
np.set_printoptions(threshold=np.inf)
import torch
import gym
import time
import algos.pytorch.a2c_9.core as core
from utils.logx import EpochLogger
import matplotlib.pyplot as plt

  
 
class PGBuffer(object):
    """
    A buffer for storing trajectories experienced by a PG agent interacting 
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda) 
    for calculating the advantages of state-action pairs.
    """
    def __init__(self, buffer_size, obs_dim, gamma=0.99, lam=0.95):
        super(PGBuffer, self).__init__()
        self.peg_mask_buf = np.zeros(shape=(buffer_size, *obs_dim), dtype=np.float32)
        self.hole_mask_buf = np.zeros(shape=(buffer_size, *obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(shape=(buffer_size), dtype=np.float32)
        self.rew_buf = np.zeros(shape=(buffer_size), dtype=np.float32)
        self.val_buf = np.zeros(shape=(buffer_size), dtype=np.float32)
        self.ret_buf = np.zeros(shape=(buffer_size), dtype=np.float32)
        self.adv_buf = np.zeros(shape=(buffer_size), dtype=np.float32)
        self.ptr, self.path_start_idx = 0, 0
        self.gamma, self.lam = gamma, lam


    def store(self, peg_mask, hole_mask, act, rew=0, val=0):
        self.peg_mask_buf[self.ptr] = peg_mask
        self.hole_mask_buf[self.ptr] = hole_mask
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
                    act=self.act_buf, ret=self.ret_buf, adv=self.adv_buf)
        return data




def a2c(venv, nenv, ac_kwargs=dict(), seed=0, 
        local_steps_per_epoch=1000, epochs=50, gamma=0.99, 
        train_v_iters=80, lam=0.97, logger_kwargs=dict(), device='cpu',
        model_path=''):
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
    ac = torch.load(model_path).to(device)


    # Set up experience buffer
    buf_list = [PGBuffer(local_steps_per_epoch, obs_dim, gamma, lam) for _ in range(nenv)]

    # Set up model saving
    logger.setup_pytorch_saver(ac)


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

            # save and log
            [buf_list[i].store(peg_mask[i],hole_mask[i],a[i],r[i],v[i]) for i in range(nenv)]
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
        
        # clear buffer
        [buf_list[i].get() for i in range(nenv)]

        # Log info about epoch
        logger.log_tabular('Epoch', epoch)
        logger.log_tabular('EpRet', with_min_and_max=True)
        logger.log_tabular('EpLen', average_only=True)
        logger.log_tabular('VVals', with_min_and_max=True)
        logger.log_tabular('TotalEnvInteracts', (epoch+1)*local_steps_per_epoch*nenv)
        logger.log_tabular('Time', time.time()-start_time)
        logger.dump_tabular()


if __name__ == '__main__':
    pass




