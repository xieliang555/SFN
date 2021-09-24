import numpy as np
import scipy.signal
from gym.spaces import Box, Discrete

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical
 
 
def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])


def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.

    input: 
        vector x, 
        [x0, 
         x1, 
         x2]

    output:
        [x0 + discount * x1 + discount^2 * x2,  
         x1 + discount * x2,
         x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]

 

class Encoder(nn.Module):
    """docstring for Encoder"""
    def __init__(self, obs_ft_dim, obs_img_dim):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1)
        self.fc = nn.Linear(1568, 256)
        self.rnn = nn.GRU(obs_ft_dim[1], 256, 1, batch_first=True)

    def forward(self, ft, img):
        _, h = self.rnn(ft)
        fea_ft = F.elu(h.squeeze(0))

        x = F.elu(self.conv1(img.unsqueeze(1)))
        x = F.elu(self.conv2(x))
        x = F.elu(self.conv3(x))
        x = F.elu(self.conv4(x))
        x = x.view(x.size(0), -1)
        fea_img = F.elu(self.fc(x))

        fea = torch.cat((fea_ft, fea_img), dim=-1)
        return fea



# class ForwardModel(nn.Module):
#     """docstring for ForwardModel"""
#     def __init__(self, observation_space, action_space):
#         super(ForwardModel, self).__init__()
#         obs_dim = observation_space.shape[-1]
#         act_dim = action_space.n
#         self.encoder = Encoder(obs_dim)
#         self.fc = nn.Linear(512, act_dim)

#     def forward(self, obs, obs_next):
#         fea_1 = self.encoder(obs)
#         fea_2 = self.encoder(obs_next)
#         y = self.fc(torch.cat((fea_1, fea_2), dim=-1))
#         return y



class A2C(nn.Module):
    """docstring for A2C"""
    def __init__(self, observation_space, action_space):
        super(A2C, self).__init__()
        obs_ft_dim = observation_space.shape[0:2]
        obs_img_dim = observation_space.shape[2:4]
        act_dim = action_space.n
        self.encoder = Encoder(obs_ft_dim, obs_img_dim)
        self.fc_pi = nn.Linear(512, act_dim)
        self.fc_v = nn.Linear(512, 1)
        
    # training network
    def forward(self, obs_ft, obs_img, act):
        fea = self.encoder(obs_ft, obs_img)
        logits = self.fc_pi(fea)
        pi = Categorical(logits=logits)
        logp = pi.log_prob(act)
        v = self.fc_v(fea).squeeze(-1)
        return logp, v

    # collecting data
    def step(self, obs_ft, obs_img):
        with torch.no_grad():
            fea = self.encoder(obs_ft, obs_img)
            logits = self.fc_pi(fea)
            pi = Categorical(logits=logits)
            a = pi.sample()
            v = self.fc_v(fea).squeeze(-1) 
        return a, v

    def act(self, obs_ft, obs_img):
        return self.step(obs_ft, obs_img)[0]







