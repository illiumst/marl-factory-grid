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
            nn.Linear(5 * 5 * 3, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 5)
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
    def __init__(self, q_net, target_q_net, env, buffer, n_steps, target_update, warmup, eps_end,
                 gamma=0.99, train_every_n_steps=4, exploration_fraction=0.2, batch_size=64, lr=1e-4, reg_weight=0.0):
        self.q_net = q_net
        self.target_q_net = target_q_net
        self.target_q_net.eval()
        self.env = env
        self.buffer = buffer
        self.n_steps = n_steps
        self.target_update = target_update
        self.warmup = warmup
        self.eps = 1.
        self.eps_end = eps_end
        self.exploration_fraction = exploration_fraction
        self.batch_size = batch_size
        self.gamma = gamma
        self.train_every_n_steps = train_every_n_steps
        self.lr = lr
        self.reg_weight = reg_weight
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=self.lr)
        self.device = 'cpu'
        self.running_reward = deque(maxlen=10)

    def to(self, device):
        self.device = device
        return self

    def anneal_eps(self, step):
        fraction = min(float(step) / int(self.exploration_fraction*self.n_steps), 1.0)
        self.eps = 1 + fraction * (self.eps_end - 1)

    def learn(self):
        step = 0
        while step < self.n_steps:
            obs, done = self.env.reset(), False
            total_reward = 0
            while not done:
                action = self.q_net.random_action() if np.random.rand() < self.eps \
                    else self.q_net.act(torch.from_numpy(obs).unsqueeze(0).float())
                next_obs, reward, done, info = env.step(action)
                print(action, reward)
                experience = Experience(obs.copy(), next_obs.copy(), action, reward, done)  # do we really need to copy?
                obs = next_obs
                self.buffer.add(experience)

                # end of step routine
                self.anneal_eps(step)
                step += 1
                total_reward += reward
                if step % self.train_every_n_steps == 0:
                    self.train()
                if step % self.target_update == 0:
                    polyak_update(self.q_net.parameters(), self.target_q_net.parameters(), 1.0)

            self.running_reward.append(total_reward)
            if step % 800 == 0:
                print(f'Step: {step} ({(step/self.n_steps)*100:.2f}%)\tRunning reward: {sum(list(self.running_reward))/len(self.running_reward)}\t eps: {self.eps:.4f}')

    def train(self):
        for _ in range(4):
            experience = self.buffer.sample(self.batch_size)
            obs = torch.stack([torch.from_numpy(e.observation) for e in experience], 0).float()
            next_obs = torch.stack([torch.from_numpy(e.next_observation) for e in experience], 0).float()
            actions = torch.tensor([e.action for e in experience]).long()
            rewards = torch.tensor([e.reward for e in experience]).float()
            dones = torch.tensor([e.done for e in experience]).float()

            with torch.no_grad():
                next_q_values = self.target_q_net(next_obs).max(-1)[0]
                target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

            q_values = self.q_net(obs).gather(-1, actions.unsqueeze(-1))

            delta = q_values - target_q_values
            loss = torch.mean(self.reg_weight * q_values + torch.pow(delta, 2))

            # Optimize the model
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 10)
            self.optimizer.step()


if __name__ == '__main__':
    from environments.factory.simple_factory import SimpleFactory, DirtProperties, MovementProperties
    dirt_props = DirtProperties(clean_amount=3, gain_amount=0.2, max_global_amount=30,
                                max_local_amount=5, spawn_frequency=1, max_spawn_ratio=0.05)
    move_props = MovementProperties(allow_diagonal_movement=True,
                                    allow_square_movement=True,
                                    allow_no_op=False)
    env = SimpleFactory(dirt_properties=dirt_props, movement_properties=move_props, n_agents=1, pomdp_radius=2, combin_agent_slices_in_obs=True, max_steps=400, omit_agent_slice_in_obs=False)
    #print(env.action_space)
    dqn, target_dqn = BaseDQN(), BaseDQN()
    learner = BaseQlearner(dqn, target_dqn, env, BaseBuffer(50000), n_steps=100000, target_update=2000, warmup=1000, train_every_n_steps=1, eps_end=0.025, reg_weight=0.0, exploration_fraction=0.3)
    print(learner.learn())
