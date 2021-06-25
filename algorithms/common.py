from typing import NamedTuple, Union
from collections import deque, OrderedDict
import numpy as np
import random
import torch
import torch.nn as nn


class BaseLearner:
    def __init__(self, env, n_agents, lr):
        self.env = env
        self.n_agents = n_agents
        self.lr = lr
        self.device = 'cpu'

    def to(self, device):
        self.device = device
        for attr, value in self.__dict__.items():
            if isinstance(value, nn.Module):
                value = value.to(self.device)
        return self


class Experience(NamedTuple):
    # can be use for a single (s_t, a, r s_{t+1}) tuple
    # or for a batch of tuples
    observation:      np.ndarray
    next_observation: np.ndarray
    action:           np.ndarray
    reward:           Union[float, np.ndarray]
    done  :           Union[bool, np.ndarray]


class BaseBuffer:
    def __init__(self, size: int):
        self.size = size
        self.experience = deque(maxlen=size)

    def __len__(self):
        return len(self.experience)

    def add(self, experience):
        self.experience.append(experience)

    def sample(self, k, cer=4):
        sample = random.choices(self.experience, k=k-cer)
        for i in range(cer): sample += [self.experience[-i]]
        observations = torch.stack([torch.from_numpy(e.observation) for e in sample], 0).float()
        next_observations = torch.stack([torch.from_numpy(e.next_observation) for e in sample], 0).float()
        actions = torch.tensor([e.action for e in sample]).long()
        rewards = torch.tensor([e.reward for e in sample]).float().view(-1, 1)
        dones = torch.tensor([e.done for e in sample]).float().view(-1, 1)
        return Experience(observations, next_observations, actions, rewards, dones)


def soft_update(local_model, target_model, tau):
    # taken from https://github.com/BY571/Munchausen-RL/blob/master/M-DQN.ipynb
    for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
        target_param.data.copy_(tau*local_param.data + (1.-tau)*target_param.data)


def mlp_maker(dims, flatten=False, activation='elu', activation_last='identity'):
    activations = {'elu': nn.ELU, 'relu': nn.ReLU,
                  'leaky_relu': nn.LeakyReLU, 'tanh': nn.Tanh,
                  'gelu': nn.GELU, 'identity': nn.Identity}
    layers = [('Flatten', nn.Flatten())] if flatten else []
    for i in range(1, len(dims)):
        layers.append((f'Layer #{i - 1}: Linear', nn.Linear(dims[i - 1], dims[i])))
        activation_str = activation if i != len(dims)-1 else activation_last
        layers.append((f'Layer #{i - 1}: {activation_str.capitalize()}', activations[activation_str]()))
    return nn.Sequential(OrderedDict(layers))



class BaseDQN(nn.Module):
    def __init__(self, dims=[3*5*5, 64, 64, 9]):
        super(BaseDQN, self).__init__()
        self.net = mlp_maker(dims, flatten=True)

    @torch.no_grad()
    def act(self, x) -> np.ndarray:
        action = self.forward(x).max(-1)[1].numpy()
        return action

    def forward(self, x):
        return self.net(x)


class BaseDDQN(BaseDQN):
    def __init__(self,
                 backbone_dims=[3*5*5, 64, 64],
                 value_dims=[64, 1],
                 advantage_dims=[64, 9]):
        super(BaseDDQN, self).__init__(backbone_dims)
        self.net = mlp_maker(backbone_dims, flatten=True)
        self.value_head         =  mlp_maker(value_dims)
        self.advantage_head     =  mlp_maker(advantage_dims)

    def forward(self, x):
        features = self.net(x)
        advantages = self.advantage_head(features)
        values = self.value_head(features)
        return values + (advantages - advantages.mean())


class QTRANtestNet(nn.Module):
    def __init__(self, backbone_dims=[3*5*5, 64, 64], q_head=[64, 9]):
        super(QTRANtestNet, self).__init__()
        self.backbone = mlp_maker(backbone_dims, flatten=True, activation_last='elu')
        self.q_head = mlp_maker(q_head)

    def forward(self, x):
        features = self.backbone(x)
        qs = self.q_head(features)
        return qs, features

    @torch.no_grad()
    def act(self, x) -> np.ndarray:
        action = self.forward(x)[0].max(-1)[1].numpy()
        return action