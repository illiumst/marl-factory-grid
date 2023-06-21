import torch
from torch.distributions import Categorical
from marl_factory_grid.algorithms.marl.iac import LoopIAC
from marl_factory_grid.algorithms.marl.base_ac import nms
from marl_factory_grid.algorithms.marl.memory import MARLActorCriticMemory


class LoopSEAC(LoopIAC):
    def __init__(self, cfg):
        super(LoopSEAC, self).__init__(cfg)

    def actor_critic(self, tm, networks, gamma, entropy_coef, vf_coef, gae_coef=0.0, **kwargs):
        obs, actions, done, reward = tm.observation, tm.action, tm.done[:, 1:], tm.reward[:, 1:]
        outputs = [net(obs, actions, tm.hidden_actor[:, 0], tm.hidden_critic[:, 0]) for net in networks]

        with torch.inference_mode(True):
            true_action_logp = torch.stack([
                torch.log_softmax(out[nms.LOGITS][ag_i, :-1], -1)
                    .gather(index=actions[ag_i, 1:, None], dim=-1)
                for ag_i, out in enumerate(outputs)
            ], 0).squeeze()

        losses = []

        for ag_i, out in enumerate(outputs):
            logits = out[nms.LOGITS][:, :-1]  # last one only needed for v_{t+1}
            critic = out[nms.CRITIC]

            entropy_loss = Categorical(logits=logits[ag_i]).entropy().mean()
            advantages = self.compute_advantages(critic, reward, done, gamma, gae_coef)

            # policy loss
            log_ap = torch.log_softmax(logits, -1)
            log_ap = torch.gather(log_ap, dim=-1, index=actions[:, 1:].unsqueeze(-1)).squeeze()

            # importance weights
            iw = (log_ap - true_action_logp).exp().detach()  # importance_weights

            a2c_loss = (-iw*log_ap * advantages.detach()).mean(-1)


            value_loss = (iw*advantages.pow(2)).mean(-1)  # n_agent

            # weighted loss
            loss = (a2c_loss + vf_coef*value_loss - entropy_coef * entropy_loss).mean()
            losses.append(loss)

        return losses

    def learn(self, tms: MARLActorCriticMemory, **kwargs):
        losses = self.actor_critic(tms, self.net, **self.cfg[nms.ALGORITHM], **kwargs)
        for ag_i, loss in enumerate(losses):
            self.optimizer[ag_i].zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.net[ag_i].parameters(), 0.5)
            self.optimizer[ag_i].step()