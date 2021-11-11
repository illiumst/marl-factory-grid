from typing import Union
import torch
import numpy as np
import pandas as pd
from algorithms.q_learner import QLearner


class VDNLearner(QLearner):
    def __init__(self, *args, **kwargs):
        super(VDNLearner, self).__init__(*args, **kwargs)
        assert self.n_agents >= 2, 'VDN requires more than one agent, use QLearner instead'

    def get_action(self, obs) -> Union[int, np.ndarray]:
        o = torch.from_numpy(obs).unsqueeze(0) if self.n_agents <= 1 else torch.from_numpy(obs)
        eps = np.random.rand(self.n_agents)
        greedy = eps > self.eps
        agent_actions = None
        actions = []
        for i in range(self.n_agents):
            if greedy[i]:
                if agent_actions is None: agent_actions = self.q_net.act(o.float())
                action = agent_actions[i]
            else:
                action = self.env.action_space.sample()
            actions.append(action)
        return np.array(actions)

    def train(self):
        if len(self.buffer) < self.batch_size: return
        for _ in range(self.n_grad_steps):
            experience = self.buffer.sample(self.batch_size, cer=self.train_every_n_steps)
            pred_q, target_q_raw = torch.zeros((self.batch_size, 1)), torch.zeros((self.batch_size, 1))
            for agent_i in range(self.n_agents):
                q_values, next_q_values_raw = self._training_routine(experience.observation[:, agent_i],
                                                                     experience.next_observation[:, agent_i],
                                                                     experience.action[:, agent_i].unsqueeze(-1))
                pred_q += q_values
                target_q_raw += next_q_values_raw
            target_q = experience.reward + (1 - experience.done) * self.gamma * target_q_raw
            loss = torch.mean(self.reg_weight * pred_q + torch.pow(pred_q - target_q, 2))
            self._backprop_loss(loss)

    def evaluate(self, n_episodes=100, render=False):
        with torch.no_grad():
            data = []
            for eval_i in range(n_episodes):
                obs, done = self.env.reset(), False
                while not done:
                    action = self.get_action(obs)
                    next_obs, reward, done, info = self.env.step(action)
                    if render: self.env.render()
                    obs = next_obs  # srsly i'm so stupid
                    info.update({'reward': reward, 'eval_episode': eval_i})
                    data.append(info)
        return pd.DataFrame(data).fillna(0)
