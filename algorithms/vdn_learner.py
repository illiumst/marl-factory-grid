import torch
from algorithms.q_learner import QLearner


class VDNLearner(QLearner):
    def __init__(self, *args, **kwargs):
        super(VDNLearner, self).__init__(*args, **kwargs)
        assert self.n_agents >= 2, 'VDN requires more than one agent, use QLearner instead'

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