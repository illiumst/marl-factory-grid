from typing import Union
import gym
import torch
import torch.nn as nn
import numpy as np
from collections import deque
from pathlib import Path
import yaml
from algorithms.common import BaseLearner, BaseBuffer, soft_update, Experience


class QLearner(BaseLearner):
    def __init__(self, q_net, target_q_net, env, buffer_size=1e5, target_update=3000, eps_end=0.05, n_agents=1,
                 gamma=0.99, train_every_n_steps=4, n_grad_steps=1, tau=1.0, max_grad_norm=10, weight_decay=1e-2,
                 exploration_fraction=0.2, batch_size=64, lr=1e-4, reg_weight=0.0, eps_start=1):
        super(QLearner, self).__init__(env, n_agents, lr)
        self.q_net = q_net
        self.target_q_net = target_q_net
        self.target_q_net.eval()
        soft_update(self.q_net, self.target_q_net, tau=1.0)
        self.buffer = BaseBuffer(buffer_size)
        self.target_update = target_update
        self.eps = eps_start
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.exploration_fraction = exploration_fraction
        self.batch_size = batch_size
        self.gamma = gamma
        self.train_every_n_steps = train_every_n_steps
        self.n_grad_steps = n_grad_steps
        self.tau = tau
        self.reg_weight = reg_weight
        self.weight_decay = weight_decay
        self.optimizer = torch.optim.AdamW(self.q_net.parameters(),
                                           lr=self.lr,
                                           weight_decay=self.weight_decay)
        self.max_grad_norm = max_grad_norm
        self.running_reward = deque(maxlen=5)
        self.running_loss = deque(maxlen=5)
        self.n_updates = 0

    def save(self, path):
        path = Path(path)  # no-op if already instance of Path
        path.mkdir(parents=True, exist_ok=True)
        hparams = {k: v for k, v in self.__dict__.items() if not(isinstance(v, BaseBuffer) or
                                                                 isinstance(v, torch.optim.Optimizer) or
                                                                 isinstance(v, gym.Env) or
                                                                 isinstance(v, nn.Module))
                   }
        hparams.update({'class': self.__class__.__name__})
        with (path / 'hparams.yaml').open('w') as outfile:
            yaml.dump(hparams, outfile)
        torch.save(self.q_net, path / 'q_net.pt')

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

                experience = Experience(observation=obs, next_observation=next_obs, action=action, reward=reward, done=done)  # do we really need to copy?
                self.buffer.add(experience)
                # end of step routine
                obs = next_obs
                step += 1
                total_reward += reward
                self.anneal_eps(step, n_steps)

                if step % self.train_every_n_steps == 0:
                    self.train()
                    self.n_updates += 1
                if step % self.target_update == 0:
                    print('UPDATE')
                    soft_update(self.q_net, self.target_q_net, tau=self.tau)

            self.running_reward.append(total_reward)
            if step % 10 == 0:
                print(f'Step: {step} ({(step/n_steps)*100:.2f}%)\tRunning reward: {sum(list(self.running_reward))/len(self.running_reward):.2f}\t'
                      f' eps: {self.eps:.4f}\tRunning loss: {sum(list(self.running_loss))/len(self.running_loss):.4f}\tUpdates:{self.n_updates}')

    def _training_routine(self, obs, next_obs, action):
        current_q_values = self.q_net(obs)
        current_q_values = torch.gather(current_q_values, dim=-1, index=action)
        next_q_values_raw = self.target_q_net(next_obs).max(dim=-1)[0].reshape(-1, 1).detach()
        return current_q_values, next_q_values_raw

    def _backprop_loss(self, loss):
        # log loss
        self.running_loss.append(loss.item())
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), self.max_grad_norm)
        self.optimizer.step()

    def train(self):
        if len(self.buffer) < self.batch_size: return
        for _ in range(self.n_grad_steps):
            experience = self.buffer.sample(self.batch_size, cer=self.train_every_n_steps)
            pred_q, target_q_raw = self._training_routine(experience.observation,
                                                          experience.next_observation,
                                                          experience.action)
            target_q = experience.reward + (1 - experience.done) * self.gamma * target_q_raw
            loss = torch.mean(self.reg_weight * pred_q + torch.pow(pred_q - target_q, 2))
            self._backprop_loss(loss)



if __name__ == '__main__':
    from environments.factory.simple_factory import SimpleFactory, DirtProperties, MovementProperties
    from algorithms.common import BaseDDQN
    from algorithms.vdn_learner import VDNLearner

    N_AGENTS = 1

    dirt_props = DirtProperties(clean_amount=3, gain_amount=0.2, max_global_amount=30,
                                max_local_amount=5, spawn_frequency=1, max_spawn_ratio=0.05)
    move_props = MovementProperties(allow_diagonal_movement=True,
                                    allow_square_movement=True,
                                    allow_no_op=False)
    env = SimpleFactory(dirt_properties=dirt_props, movement_properties=move_props, n_agents=N_AGENTS, pomdp_radius=2,  max_steps=400, omit_agent_slice_in_obs=False, combin_agent_slices_in_obs=True)

    dqn, target_dqn = BaseDDQN(), BaseDDQN()
    learner = QLearner(dqn, target_dqn, env, 40000, target_update=3500, lr=0.0007, gamma=0.99, n_agents=N_AGENTS, tau=0.95, max_grad_norm=10,
                       train_every_n_steps=4, eps_end=0.025, n_grad_steps=1, reg_weight=0.1, exploration_fraction=0.25, batch_size=64)
    #learner.save(Path(__file__).parent / 'test' / 'testexperiment1337')
    learner.learn(100000)
