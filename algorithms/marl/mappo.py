from algorithms.marl import LoopSNAC
from algorithms.marl.memory import MARLActorCriticMemory
from typing import List
import random
import torch
from torch.distributions import Categorical


class LoopMAPPO(LoopSNAC):
    def __init__(self, *args, **kwargs):
        super(LoopMAPPO, self).__init__(*args, **kwargs)

    def build_batch(self, tm: List[MARLActorCriticMemory]):
        sample = random.choices(tm, k=self.cfg['algorithm']['batch_size']-1)
        sample.append(tm[-1]) # always use latest segment in batch

        obs           = torch.cat([s.observation   for s in sample], 0)
        actions       = torch.cat([s.action        for s in sample], 0)
        hidden_actor  = torch.cat([s.hidden_actor  for s in sample], 0)
        hidden_critic = torch.cat([s.hidden_critic for s in sample], 0)
        logits        = torch.cat([s.logits        for s in sample], 0)
        values        = torch.cat([s.values        for s in sample], 0)
        reward = torch.cat([s.reward for s in sample], 0)
        done = torch.cat([s.done for s in sample], 0)


        log_props = torch.log_softmax(logits, -1)
        log_props = torch.gather(log_props, index=actions[:, 1:].unsqueeze(-1), dim=-1).squeeze()

        return obs, actions, hidden_actor, hidden_critic, log_props, values, reward, done

    def learn(self, tm: List[MARLActorCriticMemory], **kwargs):
        if len(tm) >= self.cfg['algorithm']['keep_n_segments']:
            # only learn when buffer is full
            for batch_i in range(self.cfg['algorithm']['n_updates']):
                loss = self.actor_critic(tm, self.net,  **self.cfg['algorithm'], **kwargs)
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

    def actor_critic(self, tm, network, gamma, entropy_coef, vf_coef, clip_range, gae_coef=0.0, **kwargs):
        obs, actions, hidden_actor, hidden_critic, old_log_probs, old_critic, reward, done = self.build_batch(tm)

        out = network(obs, actions, hidden_actor, hidden_critic)
        logits = out['logits'][:, :-1]  # last one only needed for v_{t+1}
        critic = out['critic']

        # monte carlo returns
        mc_returns = self.monte_carlo_returns(reward, done, gamma)
        # monte_carlo_returns = (mc_returns - mc_returns.mean()) / (mc_returns.std() + 1e-7) todo: norm across agents?
        advantages =  mc_returns - critic[:, :-1]

        # policy loss
        log_ap = torch.log_softmax(logits, -1)
        log_ap = torch.gather(log_ap, dim=-1, index=actions[:, 1:].unsqueeze(-1)).squeeze()
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
