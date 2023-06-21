from marl_factory_grid.algorithms.marl.base_ac import Names as nms
from marl_factory_grid.algorithms.marl.snac import LoopSNAC
from marl_factory_grid.algorithms.marl.memory import MARLActorCriticMemory
import torch
from torch.distributions import Categorical
from marl_factory_grid.algorithms.utils import instantiate_class


class LoopMAPPO(LoopSNAC):
    def __init__(self, *args, **kwargs):
        super(LoopMAPPO, self).__init__(*args, **kwargs)
        self.reset_memory_after_epoch = False

    def setup(self):
        self.net = instantiate_class(self.cfg[nms.AGENT])
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=3e-4, eps=1e-5)

    def learn(self, tm: MARLActorCriticMemory, **kwargs):
        if len(tm) >= self.cfg['algorithm']['buffer_size']:
            # only learn when buffer is full
            for batch_i in range(self.cfg['algorithm']['n_updates']):
                batch = tm.chunk_dataloader(chunk_len=self.cfg['algorithm']['n_steps'],
                                            k=self.cfg['algorithm']['batch_size'])
                loss = self.mappo(batch, self.net, **self.cfg[nms.ALGORITHM], **kwargs)
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), 0.5)
                self.optimizer.step()

    def monte_carlo_returns(self, rewards, done, gamma):
        rewards_ = []
        discounted_reward = torch.zeros_like(rewards[:, -1])
        for t in range(rewards.shape[1]-1, -1, -1):
            discounted_reward = rewards[:, t] + (gamma * (1.0 - done[:, t]) * discounted_reward)
            rewards_.insert(0, discounted_reward)
        rewards_ = torch.stack(rewards_, dim=1)
        return rewards_

    def mappo(self, batch, network, gamma, entropy_coef, vf_coef, clip_range, **kwargs):
        out = network(batch[nms.OBSERVATION], batch[nms.ACTION], batch[nms.HIDDEN_ACTOR], batch[nms.HIDDEN_CRITIC])
        logits = out[nms.LOGITS][:, :-1]  # last one only needed for v_{t+1}

        old_log_probs = torch.log_softmax(batch[nms.LOGITS], -1)
        old_log_probs = torch.gather(old_log_probs, index=batch[nms.ACTION][:, 1:].unsqueeze(-1), dim=-1).squeeze()

        # monte carlo returns
        mc_returns = self.monte_carlo_returns(batch[nms.REWARD], batch[nms.DONE], gamma)
        mc_returns = (mc_returns - mc_returns.mean()) / (mc_returns.std() + 1e-8) #todo: norm across agent ok?
        advantages =  mc_returns - out[nms.CRITIC][:, :-1]

        # policy loss
        log_ap = torch.log_softmax(logits, -1)
        log_ap = torch.gather(log_ap, dim=-1, index=batch[nms.ACTION][:, 1:].unsqueeze(-1)).squeeze()
        ratio = (log_ap - old_log_probs).exp()
        surr1 = ratio * advantages.detach()
        surr2 = torch.clamp(ratio, 1 - clip_range, 1 + clip_range) * advantages.detach()
        policy_loss = -torch.min(surr1, surr2).mean(-1)

        # entropy & value loss
        entropy_loss = Categorical(logits=logits).entropy().mean(-1)
        value_loss = advantages.pow(2).mean(-1)  # n_agent

        # weighted loss
        loss = policy_loss + vf_coef*value_loss - entropy_coef * entropy_loss

        return loss.mean()
