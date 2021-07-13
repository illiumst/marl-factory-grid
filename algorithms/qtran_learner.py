import torch
from algorithms.q_learner import QLearner


class QTRANLearner(QLearner):
    def __init__(self, *args, weight_opt=1., weigt_nopt=1., **kwargs):
        super(QTRANLearner, self).__init__(*args, **kwargs)
        assert self.n_agents >= 2, 'QTRANLearner requires more than one agent, use QLearner instead'
        self.weight_opt = weight_opt
        self.weigt_nopt = weigt_nopt

    def _training_routine(self, obs, next_obs, action):
        # todo remove - is inherited - only used while implementing qtran
        current_q_values = self.q_net(obs)
        current_q_values = torch.gather(current_q_values, dim=-1, index=action)
        next_q_values_raw = self.target_q_net(next_obs).max(dim=-1)[0].reshape(-1, 1).detach()
        return current_q_values, next_q_values_raw

    def local_qs(self, observations, actions):
        Q_jt = torch.zeros_like(actions)  # placeholder to sum up individual q values
        features = []
        for agent_i in range(self.n_agents):
            q_values_agent_i, features_agent_i = self.q_net(observations[:, agent_i])  # Individual action-value network
            q_values_agent_i = torch.gather(q_values_agent_i, dim=-1, index=actions[:, agent_i].unsqueeze(-1))
            Q_jt += q_values_agent_i
            features.append(features_agent_i)
        feature_sum = torch.stack(features, 0).sum(0)  # (n_agents x hdim) -> hdim
        return Q_jt

    def train(self):
        if len(self.buffer) < self.batch_size: return
        for _ in range(self.n_grad_steps):
            experience = self.buffer.sample(self.batch_size, cer=self.train_every_n_steps)

            Q_jt_prime = self.local_qs(experience.observation, experience.action)  # sum of individual q-vals
            Q_jt = None
            V_jt = None

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