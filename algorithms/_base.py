from typing import NamedTuple, Union
from collections import namedtuple, deque
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3.common.utils import polyak_update
from stable_baselines3.common.buffers import ReplayBuffer
import copy


class Experience(NamedTuple):
    observation:      np.ndarray
    next_observation: np.ndarray
    action:           np.ndarray
    reward:           Union[float, np.ndarray]
    done  :           Union[bool, np.ndarray]
    priority:         np.ndarray = 1


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
        observations = torch.stack([torch.from_numpy(e.observation) for e in sample], 0).float()
        next_observations = torch.stack([torch.from_numpy(e.next_observation) for e in sample], 0).float()
        actions = torch.tensor([e.action for e in sample]).long()
        rewards = torch.tensor([e.reward for e in sample]).float().view(-1, 1)
        dones = torch.tensor([e.done for e in sample]).float().view(-1, 1)
        return Experience(observations, next_observations, actions, rewards, dones)


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

    def act(self, x) -> np.ndarray:
        with torch.no_grad():
            action = self.net(x.view(x.shape[0], -1)).max(-1)[1].numpy()
        return action

    def forward(self, x):
        return self.net(x.view(x.shape[0], -1))

    def random_action(self):
        return random.randrange(0, 5)


class BaseQlearner:
    def __init__(self, q_net, target_q_net, env, buffer, target_update, eps_end, n_agents=1,
                 gamma=0.99, train_every_n_steps=4, n_grad_steps=1,
                 exploration_fraction=0.2, batch_size=64, lr=1e-4, reg_weight=0.0):
        self.q_net = q_net
        self.target_q_net = target_q_net
        self.q_net.apply(self.weights_init)
        self.target_q_net.eval()
        self.env = env
        self.buffer = buffer
        self.target_update = target_update
        self.eps = 1.
        self.eps_end = eps_end
        self.exploration_fraction = exploration_fraction
        self.batch_size = batch_size
        self.gamma = gamma
        self.train_every_n_steps = train_every_n_steps
        self.n_grad_steps = n_grad_steps
        self.lr = lr
        self.reg_weight = reg_weight
        self.n_agents = n_agents
        self.device = 'cpu'
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=self.lr)
        self.running_reward = deque(maxlen=10)
        self.running_loss = deque(maxlen=10)

    def to(self, device):
        self.device = device
        return self

    @staticmethod
    def weights_init(module, activation='relu'):
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            nn.init.xavier_normal_(module.weight, gain=torch.nn.init.calculate_gain(activation))
            if module.bias is not None:
                module.bias.data.fill_(0.0)

    def anneal_eps(self, step, n_steps):
        fraction = min(float(step) / int(self.exploration_fraction*n_steps), 1.0)
        self.eps = 1 + fraction * (self.eps_end - 1)

    def get_action(self, obs) -> Union[int, np.ndarray]:
        o = torch.from_numpy(obs).unsqueeze(0) if self.n_agents <= 1 else torch.from_numpy(obs)
        if np.random.rand() > self.eps:
            action = self.q_net.act(o.float())
        else:
            action = np.array([self.env.action_space.sample() for _ in range(self.n_agents)])
        return action

    def learn(self, n_steps):
        step = 0
        while step < n_steps:
            obs, done = self.env.reset(), False
            total_reward = 0
            while not done:

                action = self.get_action(obs)

                next_obs, reward, done, info = self.env.step(action if not len(action) == 1 else action[0])

                experience = Experience(observation=obs.copy(), next_observation=next_obs.copy(),
                                        action=action, reward=reward, done=done)  # do we really need to copy?
                self.buffer.add(experience)
                # end of step routine
                obs = next_obs
                step += 1
                total_reward += reward
                self.anneal_eps(step, n_steps)

                if step % self.train_every_n_steps == 0:
                    self.train()
                if step % self.target_update == 0:
                    print('UPDATE')
                    polyak_update(self.q_net.parameters(), self.target_q_net.parameters(), 1)


            self.running_reward.append(total_reward)
            if step % 10 == 0:
                print(f'Step: {step} ({(step/n_steps)*100:.2f}%)\tRunning reward: {sum(list(self.running_reward))/len(self.running_reward):.2f}\t'
                      f' eps: {self.eps:.4f}\tRunning loss: {sum(list(self.running_loss))/len(self.running_loss):.4f}')

    def _training_routine(self, obs, next_obs, action):
        current_q_values = self.q_net(obs)
        current_q_values = torch.gather(current_q_values, dim=1, index=action)
        next_q_values_raw = self.target_q_net(next_obs).max(dim=1)[0].reshape(-1, 1).detach()
        return current_q_values, next_q_values_raw

    def train(self):
        if len(self.buffer) < self.batch_size: return
        for _ in range(self.n_grad_steps):

            experience = self.buffer.sample(self.batch_size)
            #print(experience.observation.shape, experience.next_observation.shape, experience.action.shape, experience.reward.shape, experience.done.shape)
            if self.n_agents <= 1:
                pred_q, target_q_raw = self._training_routine(experience.observation, experience.next_observation, experience.action)
            else:
                pred_q, target_q_raw = torch.zeros((self.batch_size, 1)), torch.zeros((self.batch_size, 1))
                for agent_i in range(self.n_agents):
                    q_values, next_q_values_raw = self._training_routine(experience.observation[:, agent_i],
                                                                         experience.next_observation[:, agent_i],
                                                                         experience.action[:, agent_i].unsqueeze(-1)
                                                                         )
                    pred_q += q_values
                    target_q_raw += next_q_values_raw
            target_q = experience.reward  + (1 - experience.done) * self.gamma * target_q_raw
            loss = torch.mean(self.reg_weight * pred_q + torch.pow(pred_q - target_q, 2))
            #print(pred_q.shape, target_q.shape)

            # log loss
            self.running_loss.append(loss.item())
            # Optimize the model
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 10)
            self.optimizer.step()



if __name__ == '__main__':
    from environments.factory.simple_factory import SimpleFactory, DirtProperties, MovementProperties
    from algorithms.reg_dqn import RegDQN
    from stable_baselines3.common.vec_env import DummyVecEnv


    N_AGENTS = 1

    dirt_props = DirtProperties(clean_amount=3, gain_amount=0.2, max_global_amount=30,
                                max_local_amount=5, spawn_frequency=1, max_spawn_ratio=0.05)
    move_props = MovementProperties(allow_diagonal_movement=True,
                                    allow_square_movement=True,
                                    allow_no_op=False)
    env = SimpleFactory(dirt_properties=dirt_props, movement_properties=move_props, n_agents=N_AGENTS, pomdp_radius=2,  max_steps=400, omit_agent_slice_in_obs=False, combin_agent_slices_in_obs=True)
    #env = DummyVecEnv([lambda: env])
    from stable_baselines3.dqn import DQN

    #dqn = RegDQN('MlpPolicy', env, verbose=True, buffer_size = 50000, learning_starts = 64, batch_size = 64,
    #             target_update_interval = 5000, exploration_fraction = 0.25, exploration_final_eps = 0.025,
    #             train_freq=4, gradient_steps=1, reg_weight=0.05)
    #dqn.learn(100000)


    print(env.observation_space, env.action_space)
    dqn, target_dqn = BaseDQN(), BaseDQN()
    learner = BaseQlearner(dqn, target_dqn, env, BaseBuffer(5000), target_update=5000, lr=0.0001, gamma=0.99, n_agents=N_AGENTS,
                           train_every_n_steps=4, eps_end=0.05, n_grad_steps=1, reg_weight=0.05, exploration_fraction=0.25, batch_size=64)
    learner.learn(100000)
