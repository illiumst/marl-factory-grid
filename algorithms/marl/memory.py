import torch
from typing import Union, List
from torch import Tensor
import numpy as np


class ActorCriticMemory(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.__states  = []
        self.__actions = []
        self.__rewards = []
        self.__dones   = []
        self.__hiddens_actor = []
        self.__hiddens_critic = []

    def __len__(self):
        return len(self.__states)

    @property
    def observation(self):
        return torch.stack(self.__states, 0).unsqueeze(0)      # 1 x timesteps x hidden dim

    @property
    def hidden_actor(self):
        if len(self.__hiddens_actor) == 1:
            return self.__hiddens_actor[0]
        return torch.stack(self.__hiddens_actor, 0)  # layers x timesteps x hidden dim

    @property
    def hidden_critic(self):
        if len(self.__hiddens_critic) == 1:
            return self.__hiddens_critic[0]
        return torch.stack(self.__hiddens_critic, 0)  # layers x timesteps x hidden dim

    @property
    def reward(self):
        return  torch.tensor(self.__rewards).float().unsqueeze(0)  # 1 x timesteps

    @property
    def action(self):
        return torch.tensor(self.__actions).long().unsqueeze(0)  # 1 x timesteps+1

    @property
    def done(self):
        return torch.tensor(self.__dones).float().unsqueeze(0)  # 1 x timesteps

    def add_observation(self, state:  Union[Tensor, np.ndarray]):
        self.__states.append(state    if isinstance(state, Tensor) else torch.from_numpy(state))

    def add_hidden_actor(self, hidden: Tensor):
        # 1x layers x hidden dim
        if len(hidden.shape) < 3: hidden = hidden.unsqueeze(0)
        self.__hiddens_actor.append(hidden)

    def add_hidden_critic(self, hidden: Tensor):
        # 1x layers x hidden dim
        if len(hidden.shape) < 3: hidden = hidden.unsqueeze(0)
        self.__hiddens_critic.append(hidden)

    def add_action(self, action: int):
        self.__actions.append(action)

    def add_reward(self, reward: float):
        self.__rewards.append(reward)

    def add_done(self, done:   bool):
        self.__dones.append(done)

    def add(self, **kwargs):
        for k, v in kwargs.items():
            func = getattr(ActorCriticMemory, f'add_{k}')
            func(self, v)


class MARLActorCriticMemory(object):
    def __init__(self, n_agents):
        self.n_agents = n_agents
        self.memories = [
            ActorCriticMemory() for _ in range(n_agents)
        ]

    def __call__(self, agent_i):
        return self.memories[agent_i]

    def __len__(self):
        return len(self.memories[0])  # todo add assertion check!

    def reset(self):
        for mem in self.memories:
            mem.reset()

    def add(self, **kwargs):
        # todo try catch - print all possible functions
        for agent_i in range(self.n_agents):
            for k, v in kwargs.items():
                func = getattr(ActorCriticMemory, f'add_{k}')
                func(self.memories[agent_i], v[agent_i])

    @property
    def observation(self):
        all_obs = [mem.observation for mem in self.memories]
        return torch.cat(all_obs, 0)  # agents x timesteps+1 x ...

    @property
    def action(self):
        all_actions = [mem.action for mem in self.memories]
        return torch.cat(all_actions, 0)  # agents x timesteps+1 x ...

    @property
    def done(self):
        all_dones = [mem.done for mem in self.memories]
        return torch.cat(all_dones, 0).float()  # agents x timesteps x ...

    @property
    def reward(self):
        all_rewards = [mem.reward for mem in self.memories]
        return torch.cat(all_rewards, 0).float()  # agents x timesteps x ...

    @property
    def hidden_actor(self):
        all_ha = [mem.hidden_actor for mem in self.memories]
        return torch.cat(all_ha, 0)  # agents x layers x  x timesteps x hidden dim

    @property
    def hidden_critic(self):
        all_hc = [mem.hidden_critic for mem in self.memories]
        return torch.cat(all_hc, 0)  # agents  x layers x timesteps x hidden dim

