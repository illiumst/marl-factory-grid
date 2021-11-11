import numpy as np
import torch
import torch.nn.functional as F
from algorithms.q_learner import QLearner


class MQLearner(QLearner):
    # Munchhausen Q-Learning
    def __init__(self, *args, temperature=0.03, alpha=0.9, clip_l0=-1.0, **kwargs):
        super(MQLearner, self).__init__(*args, **kwargs)
        assert self.n_agents == 1, 'M-DQN currently only supports single agent training'
        self.temperature = temperature
        self.alpha = alpha
        self.clip0 = clip_l0

    def tau_ln_pi(self, qs):
        # computes log(softmax(qs/temperature))
        # Custom log-sum-exp trick from page 18 to compute the log-policy terms
        v_k = qs.max(-1)[0].unsqueeze(-1)
        advantage = qs - v_k
        logsum = torch.logsumexp(advantage / self.temperature, -1).unsqueeze(-1)
        tau_ln_pi = advantage - self.temperature * logsum
        return tau_ln_pi

    def train(self):
        if len(self.buffer) < self.batch_size: return
        for _ in range(self.n_grad_steps):

            experience = self.buffer.sample(self.batch_size, cer=self.train_every[-1])

            with torch.no_grad():
                q_target_next = self.target_q_net(experience.next_observation)
                tau_log_pi_next = self.tau_ln_pi(q_target_next)

                q_k_targets = self.target_q_net(experience.observation)
                log_pi = self.tau_ln_pi(q_k_targets)

                pi_target = F.softmax(q_target_next / self.temperature, dim=-1)
                q_target = (self.gamma * (pi_target * (q_target_next - tau_log_pi_next) * (1 - experience.done)).sum(-1)).unsqueeze(-1)

                munchausen_addon = log_pi.gather(-1, experience.action)

                munchausen_reward = (experience.reward + self.alpha * torch.clamp(munchausen_addon, min=self.clip0, max=0))

                # Compute Q targets for current states
                m_q_target = munchausen_reward + q_target

            # Get expected Q values from local model
            q_k = self.q_net(experience.observation)
            pred_q = q_k.gather(-1, experience.action)

            # Compute loss
            loss = torch.mean(self.reg_weight * pred_q + torch.pow(pred_q - m_q_target, 2))
            self._backprop_loss(loss)

from tqdm import trange
from collections import deque
class MQICMLearner(MQLearner):
    def __init__(self, *args, icm, **kwargs):
        super(MQICMLearner, self).__init__(*args, **kwargs)
        self.icm = icm
        self.icm_optimizer = torch.optim.AdamW(self.icm.parameters())
        self.normalize_reward = deque(maxlen=1000)

    def on_all_done(self):
        from collections import deque
        losses = deque(maxlen=100)
        for b in trange(10000):
            batch = self.buffer.sample(128, 0)
            s0, s1, a = batch.observation,  batch.next_observation, batch.action
            loss = self.icm(s0, s1, a.squeeze())['loss']
            self.icm_optimizer.zero_grad()
            loss.backward()
            self.icm_optimizer.step()
            losses.append(loss.item())
            if b%100 == 0:
                print(np.mean(losses))
