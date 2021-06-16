from typing import Tuple, NamedTuple
from collections import namedtuple, deque
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3.common.utils import polyak_update


class Experience(NamedTuple):
    observation: np.ndarray
    next_observation: np.ndarray
    action:      int
    reward:      float
    done  :      bool
    priority:    float = 1
    info  :      dict = {}


class BaseBuffer:
    def __init__(self, size: int):
        self.size = size
        self.experience = deque(maxlen=size)

    def __len__(self):
        return len(self.experience)

    def add(self, experience):
        self.experience.append(experience)

    def sample(self, k):
        sample = random.choices(self.experience, k=k)
        return sample


class PERBuffer(BaseBuffer):
    def __init__(self, size, alpha=0.2):
        super(PERBuffer, self).__init__(size)
        self.alpha = alpha

    def sample(self, k):
        pr = [abs(e.priority)**self.alpha for e in self.experience]
        pr = np.array(pr) / sum(pr)
        idxs = random.choices(range(len(self)), weights=pr, k=k)
        pass


class BaseDQN(nn.Module):
    def __init__(self):
        super(BaseDQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(3*5*5, 64),
            nn.ReLU(),
            nn.Linear(64,  64),
            nn.ReLU(),
            nn.Linear(64, 9)
        )

    def act(self, x):
        with torch.no_grad():
            action = self.net(x.view(x.shape[0], -1)).argmax(-1).item()
        return action

    def forward(self, x):
        return self.net(x.view(x.shape[0], -1))

    def random_action(self):
        return random.randrange(0, 5)


class BaseQlearner:
    def __init__(self, q_net, target_q_net, env, buffer, target_update, warmup, eps_end,
                 gamma=0.99, train_every_n_steps=4, n_grad_steps=1,
                 exploration_fraction=0.2, batch_size=64, lr=1e-4, reg_weight=0.0):
        self.q_net = q_net
        self.target_q_net = target_q_net
        self.target_q_net.load_state_dict(self.q_net.state_dict())
        self.target_q_net.eval()
        self.env = env
        self.buffer = buffer
        self.target_update = target_update
        self.warmup = warmup
        self.eps = 1.
        self.eps_end = eps_end
        self.exploration_fraction = exploration_fraction
        self.batch_size = batch_size
        self.gamma = gamma
        self.train_every_n_steps = train_every_n_steps
        self.n_grad_steps = n_grad_steps
        self.lr = lr
        self.reg_weight = reg_weight
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=self.lr)
        self.device = 'cpu'
        self.running_reward = deque(maxlen=10)
        self.running_loss = deque(maxlen=10)

    def to(self, device):
        self.device = device
        return self

    def anneal_eps(self, step, n_steps):
        fraction = min(float(step) / int(self.exploration_fraction*n_steps), 1.0)
        eps = 1 + fraction * (self.eps_end - 1)
        return eps

    def learn(self, n_steps):
        step, eps = 0, 1
        while step < n_steps:
            obs, done = self.env.reset(), False
            total_reward = 0
            while not done:

                action = self.q_net.act(torch.from_numpy(obs).unsqueeze(0).float()) \
                    if np.random.rand() > eps else env.action_space.sample()

                next_obs, reward, done, info = env.step(action)

                experience = Experience(observation=obs, next_observation=next_obs, action=action, reward=reward, done=done)  # do we really need to copy?
                self.buffer.add(experience)
                # end of step routine
                obs = next_obs
                step += 1
                total_reward += reward
                eps = self.anneal_eps(step, n_steps)

                if step % self.train_every_n_steps == 0:
                    self.train()
                if step % self.target_update == 0:
                    self.target_q_net.load_state_dict(self.q_net.state_dict())


            self.running_reward.append(total_reward)
            if step % 10 == 0:
                print(f'Step: {step} ({(step/n_steps)*100:.2f}%)\tRunning reward: {sum(list(self.running_reward))/len(self.running_reward)}\t'
                      f' eps: {eps:.4f}\tRunning loss: {sum(list(self.running_loss))/len(self.running_loss)}')


    def train(self):
        for _ in range(self.n_grad_steps):
            experience = self.buffer.sample(self.batch_size)

            obs = torch.stack([torch.from_numpy(e.observation) for e in experience], 0).float()
            next_obs = torch.stack([torch.from_numpy(e.next_observation) for e in experience], 0).float()
            actions = torch.tensor([e.action for e in experience]).long()
            rewards = torch.tensor([e.reward for e in experience]).float()
            dones = torch.tensor([e.done for e in experience]).float()

            next_q_values = self.target_q_net(next_obs).detach().max(-1)[0]
            target_q_values = rewards + (1. - dones) * self.gamma * next_q_values


            q_values = self.q_net(obs).gather(-1, actions.unsqueeze(0))

            delta = q_values - target_q_values
            loss = torch.mean(self.reg_weight * q_values + torch.pow(delta, 2))

            self.running_loss.append(loss.item())

            # Optimize the model
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 10)
            self.optimizer.step()


if __name__ == '__main__':
    from environments.factory.simple_factory import SimpleFactory, DirtProperties, MovementProperties
    from algorithms.reg_dqn import RegDQN
    dirt_props = DirtProperties(clean_amount=3, gain_amount=0.2, max_global_amount=30,
                                max_local_amount=5, spawn_frequency=1, max_spawn_ratio=0.05)
    move_props = MovementProperties(allow_diagonal_movement=True,
                                    allow_square_movement=True,
                                    allow_no_op=False)
    env = SimpleFactory(dirt_properties=dirt_props, movement_properties=move_props, n_agents=1, pomdp_radius=2,  max_steps=400, omit_agent_slice_in_obs=False)
    #print(env.action_space)
    from stable_baselines3.dqn import DQN

    #dqn = RegDQN('MlpPolicy', env, verbose=True, buffer_size = 50000, learning_starts = 25000, batch_size = 64, target_update_interval = 5000, exploration_fraction = 0.25, exploration_final_eps = 0.025)
    #print(dqn.policy)
    #dqn.learn(100000)


    print(env.observation_space, env.action_space)
    dqn, target_dqn = BaseDQN(), BaseDQN()
    learner = BaseQlearner(dqn, target_dqn, env, BaseBuffer(50000), target_update=5000, warmup=25000, lr=1e-4, gamma=0.99,
                           train_every_n_steps=4, eps_end=0.05, reg_weight=0.1, exploration_fraction=0.25, batch_size=64)
    learner.learn(100000)
