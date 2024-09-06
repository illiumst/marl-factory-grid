import numpy as np
from collections import deque
import torch
from typing import Union
from torch import Tensor
from torch.utils.data import Dataset, ConcatDataset
import random


class ActorCriticMemory(object):
    def __init__(self, capacity=10):
        self.capacity = capacity
        self.reset()

    def reset(self):
        self.__actions        = LazyTensorFiFoQueue(maxlen=self.capacity+1)
        self.__hidden_actor   = LazyTensorFiFoQueue(maxlen=self.capacity+1)
        self.__hidden_critic  = LazyTensorFiFoQueue(maxlen=self.capacity+1)
        self.__states         = LazyTensorFiFoQueue(maxlen=self.capacity+1)
        self.__rewards        = LazyTensorFiFoQueue(maxlen=self.capacity+1)
        self.__dones          = LazyTensorFiFoQueue(maxlen=self.capacity+1)
        self.__logits         = LazyTensorFiFoQueue(maxlen=self.capacity+1)
        self.__values         = LazyTensorFiFoQueue(maxlen=self.capacity+1)

    def __len__(self):
        return len(self.__rewards) - 1

    @property
    def observation(self, sls=slice(0, None)):  # add time dimension through stacking
        return self.__states[sls].unsqueeze(0)      # 1 x time x hidden dim

    @property
    def hidden_actor(self,  sls=slice(0, None)):  # 1 x n_layers x dim
        return self.__hidden_actor[sls].unsqueeze(0)    # 1 x time x n_layers x dim

    @property
    def hidden_critic(self, sls=slice(0, None)):  # 1 x n_layers x dim
        return self.__hidden_critic[sls].unsqueeze(0)    # 1 x time x n_layers x dim

    @property
    def reward(self, sls=slice(0, None)):
        return self.__rewards[sls].squeeze().unsqueeze(0)  # 1 x time

    @property
    def action(self, sls=slice(0, None)):
        return self.__actions[sls].long().squeeze().unsqueeze(0)  # 1 x time

    @property
    def done(self, sls=slice(0, None)):
        return self.__dones[sls].float().squeeze().unsqueeze(0)  # 1 x time

    @property
    def logits(self, sls=slice(0, None)):  # assumes a trailing 1 for time dimension - common when using output from NN
        return self.__logits[sls].squeeze().unsqueeze(0)  # 1 x time x actions

    @property
    def values(self, sls=slice(0, None)):
        return self.__values[sls].squeeze().unsqueeze(0)  # 1 x time x actions

    def add_observation(self, state:  Union[Tensor, np.ndarray]):
        self.__states.append(state    if isinstance(state, Tensor) else torch.from_numpy(state))

    def add_hidden_actor(self, hidden: Tensor):
        # layers x hidden dim
        self.__hidden_actor.append(hidden)

    def add_hidden_critic(self, hidden: Tensor):
        # layers x hidden dim
        self.__hidden_critic.append(hidden)

    def add_action(self, action: Union[int, Tensor]):
        if not isinstance(action, Tensor):
            action = torch.tensor(action)
        self.__actions.append(action)

    def add_reward(self, reward: Union[float, Tensor]):
        if not isinstance(reward, Tensor):
            reward = torch.tensor(reward)
        self.__rewards.append(reward)

    def add_done(self, done:   bool):
        if not isinstance(done, Tensor):
            done = torch.tensor(done)
        self.__dones.append(done)

    def add_logits(self, logits: Tensor):
        self.__logits.append(logits)

    def add_values(self, values: Tensor):
        self.__values.append(values)

    def add(self, **kwargs):
        for k, v in kwargs.items():
            func = getattr(ActorCriticMemory, f'add_{k}')
            func(self, v)


class MARLActorCriticMemory(object):
    def __init__(self, n_agents, capacity):
        self.n_agents = n_agents
        self.memories = [
            ActorCriticMemory(capacity) for _ in range(n_agents)
        ]

    def __call__(self, agent_i):
        return self.memories[agent_i]

    def __len__(self):
        return len(self.memories[0])  # todo add assertion check!

    def reset(self):
        for mem in self.memories:
            mem.reset()

    def add(self, **kwargs):
        for agent_i in range(self.n_agents):
            for k, v in kwargs.items():
                func = getattr(ActorCriticMemory, f'add_{k}')
                func(self.memories[agent_i], v[agent_i])

    def __getattr__(self, attr):
        all_attrs = [getattr(mem, attr) for mem in self.memories]
        return torch.cat(all_attrs, 0)  # agent x time ...

    def chunk_dataloader(self, chunk_len, k):
        datasets = [ExperienceChunks(mem, chunk_len, k) for mem in self.memories]
        dataset = ConcatDataset(datasets)
        data = [dataset[i] for i in range(len(dataset))]
        data = custom_collate_fn(data)
        return data


def custom_collate_fn(batch):
    elem = batch[0]
    return {key: torch.cat([d[key] for d in batch], dim=0) for key in elem}


class ExperienceChunks(Dataset):
    def __init__(self, memory, chunk_len, k):
        assert chunk_len <= len(memory), 'chunk_len cannot be longer than the size of the memory'
        self.memory = memory
        self.chunk_len = chunk_len
        self.k = k

    @property
    def whitelist(self):
        whitelist = torch.ones(len(self.memory) - self.chunk_len)
        for d in self.memory.done.squeeze().nonzero().flatten():
            whitelist[max((0, d-self.chunk_len-1)):d+2] = 0
        whitelist[0] = 0
        return whitelist.tolist()

    def sample(self, start=1):
        cl = self.chunk_len
        sample = dict(observation=self.memory.observation[:, start:start+cl+1],
                      action=self.memory.action[:, start-1:start+cl],
                      hidden_actor=self.memory.hidden_actor[:, start-1],
                      hidden_critic=self.memory.hidden_critic[:, start-1],
                      reward=self.memory.reward[:, start:start + cl],
                      done=self.memory.done[:, start:start + cl],
                      logits=self.memory.logits[:, start:start + cl],
                      values=self.memory.values[:, start:start + cl])
        return sample

    def __len__(self):
        return self.k

    def __getitem__(self, i):
        idx = random.choices(range(0, len(self.memory) - self.chunk_len), weights=self.whitelist, k=1)
        return self.sample(idx[0])


class LazyTensorFiFoQueue:
    def __init__(self, maxlen):
        self.maxlen = maxlen
        self.reset()

    def reset(self):
        self.__lazy_queue = deque(maxlen=self.maxlen)
        self.shape = None
        self.queue = None

    def shape_init(self, tensor: Tensor):
        self.shape = torch.Size([self.maxlen, *tensor.shape])

    def build_tensor_queue(self):
        if len(self.__lazy_queue) > 0:
            block = torch.stack(list(self.__lazy_queue), dim=0)
            l = block.shape[0]
            if self.queue is None:
                self.queue = block
            elif self.true_len() <= self.maxlen:
                self.queue = torch.cat((self.queue, block),  dim=0)
            else:
                self.queue = torch.cat((self.queue[l:], block),  dim=0)
            self.__lazy_queue.clear()

    def append(self, data):
        if self.shape is None:
            self.shape_init(data)
        self.__lazy_queue.append(data)
        if len(self.__lazy_queue) >= self.maxlen:
            self.build_tensor_queue()

    def true_len(self):
        return len(self.__lazy_queue) + (0 if self.queue is None else self.queue.shape[0])

    def __len__(self):
        return min((self.true_len(), self.maxlen))

    def __str__(self):
        return f'LazyTensorFiFoQueue\tmaxlen: {self.maxlen}, shape: {self.shape}, ' \
               f'len: {len(self)}, true_len: {self.true_len()}, elements in lazy queue: {len(self.__lazy_queue)}'

    def __getitem__(self, item_or_slice):
        self.build_tensor_queue()
        return self.queue[item_or_slice]




