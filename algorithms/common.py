from typing import NamedTuple, Union
from collections import deque, OrderedDict, defaultdict
import numpy as np
import random
import torch
import torch.nn as nn


class Experience(NamedTuple):
    # can be use for a single (s_t, a, r s_{t+1}) tuple
    # or for a batch of tuples
    observation:      np.ndarray
    next_observation: np.ndarray
    action:           np.ndarray
    reward:           Union[float, np.ndarray]
    done  :           Union[bool, np.ndarray]
    episode:          int = -1


class BaseLearner:
    def __init__(self, env, n_agents=1, train_every=('step', 4), n_grad_steps=1, stack_n_frames=1):
        assert train_every[0] in ['step', 'episode'], 'train_every[0] must be one of ["step", "episode"]'
        self.env = env
        self.n_agents = n_agents
        self.n_grad_steps = n_grad_steps
        self.train_every = train_every
        self.stack_n_frames = deque(maxlen=stack_n_frames)
        self.device = 'cpu'
        self.n_updates = 0
        self.step = 0
        self.episode_step = 0
        self.episode = 0
        self.running_reward = deque(maxlen=5)

    def to(self, device):
        self.device = device
        for attr, value in self.__dict__.items():
            if isinstance(value, nn.Module):
                value = value.to(self.device)
        return self

    def get_action(self, obs) -> Union[int, np.ndarray]:
        pass

    def on_new_experience(self, experience):
        pass

    def on_step_end(self, n_steps):
        pass

    def on_episode_end(self, n_steps):
        pass

    def on_all_done(self):
        pass

    def train(self):
        pass

    def learn(self, n_steps):
        train_type, train_freq = self.train_every
        while self.step < n_steps:
            obs, done = self.env.reset(), False
            total_reward = 0
            self.episode_step = 0
            while not done:

                action = self.get_action(obs)

                next_obs, reward, done, info = self.env.step(action if not len(action) == 1 else action[0])

                experience = Experience(observation=obs, next_observation=next_obs,
                                        action=action, reward=reward,
                                        done=done, episode=self.episode)  # do we really need to copy?
                self.on_new_experience(experience)
                # end of step routine
                obs = next_obs
                total_reward += reward
                self.step += 1
                self.episode_step += 1
                self.on_step_end(n_steps)
                if train_type == 'step' and (self.step % train_freq == 0):
                    self.train()
                    self.n_updates += 1
            self.on_episode_end(n_steps)
            if train_type == 'episode' and (self.episode % train_freq == 0):
                self.train()
                self.n_updates += 1

            self.running_reward.append(total_reward)
            self.episode += 1
            try:
                if self.step % 10 == 0:
                    print(
                        f'Step: {self.step} ({(self.step / n_steps) * 100:.2f}%)\tRunning reward: {sum(list(self.running_reward)) / len(self.running_reward):.2f}\t'
                        f' eps: {self.eps:.4f}\tRunning loss: {sum(list(self.running_loss)) / len(self.running_loss):.4f}\tUpdates:{self.n_updates}')
            except Exception as e:
                pass
        self.on_all_done()


class BaseBuffer:
    def __init__(self, size: int):
        self.size = size
        self.experience = deque(maxlen=size)

    def __len__(self):
        return len(self.experience)

    def add(self, exp: Experience):
        self.experience.append(exp)

    def sample(self, k, cer=4):
        sample = random.choices(self.experience, k=k-cer)
        for i in range(cer): sample += [self.experience[-i]]
        observations = torch.stack([torch.from_numpy(e.observation) for e in sample], 0).float()
        next_observations = torch.stack([torch.from_numpy(e.next_observation) for e in sample], 0).float()
        actions = torch.tensor([e.action for e in sample]).long()
        rewards = torch.tensor([e.reward for e in sample]).float().view(-1, 1)
        dones = torch.tensor([e.done for e in sample]).float().view(-1, 1)
        #print(observations.shape, next_observations.shape, actions.shape, rewards.shape, dones.shape)
        return Experience(observations, next_observations, actions, rewards, dones)


class TrajectoryBuffer(BaseBuffer):
    def __init__(self, size):
        super(TrajectoryBuffer, self).__init__(size)
        self.experience = defaultdict(list)

    def add(self, exp: Experience):
        self.experience[exp.episode].append(exp)
        if len(self.experience) > self.size:
            oldest_traj_key = list(sorted(self.experience.keys()))[0]
            del self.experience[oldest_traj_key]


def soft_update(local_model, target_model, tau):
    # taken from https://github.com/BY571/Munchausen-RL/blob/master/M-DQN.ipynb
    for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
        target_param.data.copy_(tau*local_param.data + (1.-tau)*target_param.data)


def mlp_maker(dims, flatten=False, activation='elu', activation_last='identity'):
    activations = {'elu': nn.ELU, 'relu': nn.ReLU, 'sigmoid': nn.Sigmoid,
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
                 advantage_dims=[64, 9],
                 activation='elu'):
        super(BaseDDQN, self).__init__(backbone_dims)
        self.net = mlp_maker(backbone_dims, activation=activation, flatten=True)
        self.value_head         =  mlp_maker(value_dims)
        self.advantage_head     =  mlp_maker(advantage_dims)

    def forward(self, x):
        features = self.net(x)
        advantages = self.advantage_head(features)
        values = self.value_head(features)
        return values + (advantages - advantages.mean())


class BaseICM(nn.Module):
    def __init__(self, backbone_dims=[2*3*5*5, 64, 64], head_dims=[2*64, 64, 9]):
        super(BaseICM, self).__init__()
        self.backbone = mlp_maker(backbone_dims, flatten=True)
        self.icm = mlp_maker(head_dims)
        self.ce = nn.CrossEntropyLoss()

    def forward(self, s0, s1, a):
        phi_s0 = self.backbone(s0)
        phi_s1 = self.backbone(s1)
        cat = torch.cat((phi_s0, phi_s1), dim=1)
        a_prime = torch.softmax(self.icm(cat), dim=-1)
        ce = self.ce(a_prime, a)
        return dict(prediction=a_prime, loss=ce)