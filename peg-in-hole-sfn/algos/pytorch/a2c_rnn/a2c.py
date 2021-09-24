import numpy as np
import torch
from torch.optim import Adam
import gym
import time
import algos.pytorch.a2c_rnn.core as core
from utils.logx import EpochLogger

 

class PGBuffer(object):
    """
    A buffer for storing trajectories experienced by a PG agent interacting 
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda) 
    for calculating the advantages of state-action pairs.
    """
    def __init__(self, buffer_size, obs_dim, gamma=0.99, lam=0.95):
        super(PGBuffer, self).__init__()
        self.obs_buf = np.zeros(shape=(buffer_size, *obs_dim), dtype=np.float32)
        self.obs_next_buf = np.zeros(shape=(buffer_size, *obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(shape=(buffer_size), dtype=np.uint8)
        self.rew_buf = np.zeros(shape=(buffer_size), dtype=np.float32)
        self.val_buf = np.zeros(shape=(buffer_size), dtype=np.float32)
        self.ret_buf = np.zeros(shape=(buffer_size), dtype=np.float32)
        self.adv_buf = np.zeros(shape=(buffer_size), dtype=np.float32)
        self.ptr, self.path_start_idx = 0, 0
        self.gamma, self.lam = gamma, lam


    def store(self, obs, act, rew=0, val=0, obs_next=0):
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.obs_next_buf[self.ptr] = obs_next
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
        data = dict(obs=self.obs_buf, act=self.act_buf, 
                    ret=self.ret_buf, adv=self.adv_buf, 
                    obs_next = self.obs_next_buf)
        return data



def a2c(venv, nenv, actor_critic=core.MLPActorCritic, ac_kwargs=dict(), seed=0, 
        local_steps_per_epoch=1000, epochs=50, gamma=0.99, pi_lr=3e-4, vf_lr=1e-3, 
        train_v_iters=80, lam=0.97, logger_kwargs=dict(), save_freq=10, device='cpu'):
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
    obs_dim = venv.observation_space.shape
    act_dim = venv.action_space.shape

    # Create actor-critic module
    ac = actor_critic(venv.observation_space, venv.action_space, **ac_kwargs).to(device)

    # Count variables
    var_counts = tuple(core.count_vars(module) for module in [ac.pi, ac.v])
    logger.log('\nNumber of parameters: \t pi: %d, \t v: %d\n'%var_counts)

    # Set up experience buffer
    buf_list = [PGBuffer(local_steps_per_epoch, obs_dim, gamma, lam) for _ in range(nenv)]

    # Set up function for computing VPG policy loss
    def compute_loss_pi(data):
        obs, act, adv = data['obs'], data['act'], data['adv']

        # Policy loss
        pi, logp = ac.pi(obs, act)
        loss_pi = -(logp * adv).mean()
        return loss_pi

    # Set up function for computing value loss
    def compute_loss_v(data):
        obs, ret = data['obs'], data['ret']
        return ((ac.v(obs) - ret)**2).mean()

    # Set up optimizers for policy and value function
    pi_optimizer = Adam(ac.pi.parameters(), lr=pi_lr)
    vf_optimizer = Adam(ac.v.parameters(), lr=vf_lr)

    # Set up model saving
    logger.setup_pytorch_saver(ac)

    def update():
        obs = torch.cat([torch.tensor(
            buf_list[i].get()['obs']) for i in range(nenv)])
        act = torch.cat([torch.tensor(
            buf_list[i].get()['act'], dtype=torch.long) for i in range(nenv)])
        adv = torch.cat([torch.tensor(
            buf_list[i].get()['adv']) for i in range(nenv)])
        adv = (adv - torch.mean(adv)) / torch.std(adv)
        ret = torch.cat([torch.tensor(
            buf_list[i].get()['ret']) for i in range(nenv)])
        obs_next = torch.cat([torch.tensor(
            buf_list[i].get()['obs_next']) for i in range(nenv)])
        data = {'obs':obs.to(device), 
                'act':act.to(device), 
                'adv':adv.to(device), 
                'ret':ret.to(device), 
                'obs_next':obs_next.to(device)}

        # Get loss and info values before update
        pi_l_old = compute_loss_pi(data)
        pi_l_old = pi_l_old.item()
        v_l_old = compute_loss_v(data).item()

        # Train policy with a single step of gradient descent
        pi_optimizer.zero_grad()
        loss_pi = compute_loss_pi(data)
        loss_pi.backward()
        pi_optimizer.step()

        # Value function learning
        for i in range(train_v_iters):
            vf_optimizer.zero_grad()
            loss_v = compute_loss_v(data)
            loss_v.backward()
            vf_optimizer.step()

        # Log changes from update
        logger.store(LossPi=pi_l_old, LossV=v_l_old,
                     DeltaLossPi=(loss_pi.item() - pi_l_old),
                     DeltaLossV=(loss_v.item() - v_l_old))

    # Prepare for interaction with environment
    start_time = time.time()

    # Main loop: collect experience in env and update/log each epoch
    for epoch in range(epochs):
        o, ep_ret, ep_len = venv.reset(), np.zeros(nenv), np.zeros(nenv)

        for t in range(local_steps_per_epoch):
            a, v, logp = ac.step(torch.as_tensor(o, dtype=torch.float32).cpu().to(device))

            next_o, r, d, info = venv.step(a)

            ep_ret += r
            ep_len += 1

            # save and log
            [buf_list[i].store(o[i],a[i],r[i],v[i],next_o[i]) for i in range(nenv)]
            logger.store(VVals=np.mean(v))
            
            # Update obs (critical!)
            o = next_o
            
            if t == local_steps_per_epoch-1:
                _, last_val, _ = ac.step(torch.as_tensor(o, dtype=torch.float32).cpu().to(device))
                [buf_list[i].finish_path(last_val[i]) for i in range(nenv)]
            elif d.any():
                # print('info', info)
                # replace with for (i,done) in enumerate(d), 参考VecFrameStack
                ob_next = np.stack([info[i]['ob_next'] if info[i] else np.zeros_like(o[0]) for i in range(len(info))], axis=0)
                _, last_val, _ = ac.step(torch.as_tensor(ob_next, dtype=torch.float32).cpu().to(device))
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

        # Perform VPG update!
        update()

        # Log info about epoch
        logger.log_tabular('Epoch', epoch)
        logger.log_tabular('EpRet', with_min_and_max=True)
        logger.log_tabular('EpLen', average_only=True)
        logger.log_tabular('VVals', with_min_and_max=True)
        logger.log_tabular('TotalEnvInteracts', (epoch+1)*local_steps_per_epoch*nenv)
        logger.log_tabular('LossPi', average_only=True)
        logger.log_tabular('LossV', average_only=True)
        logger.log_tabular('DeltaLossPi', average_only=True)
        logger.log_tabular('DeltaLossV', average_only=True)
        logger.log_tabular('Time', time.time()-start_time)
        logger.dump_tabular()


if __name__ == '__main__':
    pass