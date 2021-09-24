import numpy as np
import scipy.signal
from gym.spaces import Box, Discrete

import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical

 
def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


# def mlp(sizes, activation, output_activation=nn.Identity):
#     layers = []
#     for j in range(len(sizes)-1):
#         act = activation if j < len(sizes)-2 else output_activation
#         layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
#     return nn.Sequential(*layers)
 
 
class mlp(nn.Module):
    """docstring for mlp"""
    def __init__(self, sizes, activation, output_activation=nn.Identity):
        super(mlp, self).__init__()

        self.rnn = nn.GRU(sizes[0], sizes[1], 1, batch_first=True)
        self.relu = nn.ReLU()
        layers = []
        for j in range(1, len(sizes)-1):
            act = activation if j < len(sizes)-2 else output_activation
            layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
        self.mlp_fc = nn.Sequential(*layers)
        
    def forward(self, x):
        _, h = self.rnn(x)
        h = self.relu(h.squeeze(0))
        return self.mlp_fc(h)



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
    def __init__(self, obs_dim):
        super(Encoder, self).__init__()
        self.rnn = nn.GRU(obs_dim, 256, 1, batch_first=True)
        self.relu = nn.ReLU()

    def forward(self, x):
        _, h = self.rnn(x)
        y = self.relu(h.squeeze(0))
        return y



class ForwardModel(nn.Module):
    """docstring for ForwardModel"""
    def __init__(self, observation_space, action_space):
        super(ForwardModel, self).__init__()
        obs_dim = observation_space.shape[-1]
        act_dim = action_space.n
        self.encoder = Encoder(obs_dim)
        self.fc = nn.Linear(512, act_dim)

    def forward(self, obs, obs_next):
        fea_1 = self.encoder(obs)
        fea_2 = self.encoder(obs_next)
        y = self.fc(torch.cat((fea_1, fea_2), dim=-1))
        return y



class A2C(nn.Module):
    """docstring for A2C"""
    def __init__(self, observation_space, action_space):
        super(A2C, self).__init__()
        obs_dim = observation_space.shape[-1]
        act_dim = action_space.n
        self.encoder = Encoder(obs_dim)
        self.fc_pi = nn.Linear(256, act_dim)
        self.fc_v = nn.Linear(256, 1)
        # fix the feature encoder
        # self.encoder.requires_grad_(requires_grad=False)
        
    # training network
    def forward(self, obs, act):
        fea = self.encoder(obs)
        logits = self.fc_pi(fea)
        pi = Categorical(logits=logits)
        logp = pi.log_prob(act)
        v = self.fc_v(fea).squeeze(-1)
        return logp, v

    # collecting data
    def step(self, obs):
        with torch.no_grad():
            fea = self.encoder(obs)
            logits = self.fc_pi(fea)
            pi = Categorical(logits=logits)
            a = pi.sample()
            v = self.fc_v(fea).squeeze(-1) 
        return a, v

    def act(self, obs):
        return self.step(obs)[0]


 












        




