import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.nn.utils import spectral_norm


class RecurrentAC(nn.Module):
    def __init__(self, observation_size, n_actions, obs_emb_size,
                 action_emb_size, hidden_size_actor, hidden_size_critic,
                 n_agents, use_agent_embedding=True):
        super(RecurrentAC, self).__init__()
        observation_size = np.prod(observation_size)
        self.n_layers = 1
        self.n_actions = n_actions
        self.use_agent_embedding = use_agent_embedding
        self.hidden_size_actor = hidden_size_actor
        self.hidden_size_critic = hidden_size_critic
        self.action_emb_size    = action_emb_size
        self.obs_proj   = nn.Linear(observation_size, obs_emb_size)
        self.action_emb =  nn.Embedding(n_actions+1, action_emb_size, padding_idx=0)
        self.agent_emb  =  nn.Embedding(n_agents, action_emb_size)
        mix_in_size = obs_emb_size+action_emb_size if not use_agent_embedding else obs_emb_size+n_agents*action_emb_size
        self.mix = nn.Sequential(nn.Tanh(),
                                 nn.Linear(mix_in_size, obs_emb_size),
                                 nn.Tanh(),
                                 nn.Linear(obs_emb_size, obs_emb_size)
                                 )
        self.gru_actor   = nn.GRU(obs_emb_size, hidden_size_actor,  batch_first=True, num_layers=self.n_layers)
        self.gru_critic  = nn.GRU(obs_emb_size, hidden_size_critic, batch_first=True, num_layers=self.n_layers)
        self.action_head = nn.Sequential(
            nn.Linear(hidden_size_actor, hidden_size_actor),
            nn.Tanh(),
            nn.Linear(hidden_size_actor, n_actions)
        )
        #            spectral_norm(nn.Linear(hidden_size_actor, hidden_size_actor)),
        self.critic_head = nn.Sequential(
            nn.Linear(hidden_size_critic, hidden_size_critic),
            nn.Tanh(),
            nn.Linear(hidden_size_critic, 1)
        )
        #self.action_head[-1].weight.data.uniform_(-3e-3, 3e-3)
        #self.action_head[-1].bias.data.uniform_(-3e-3, 3e-3)

    def init_hidden_actor(self):
        return torch.zeros(1, self.n_layers, self.hidden_size_actor)

    def init_hidden_critic(self):
        return torch.zeros(1, self.n_layers, self.hidden_size_critic)

    def forward(self, observations, actions, hidden_actor=None, hidden_critic=None):
        n_agents, t, *_ = observations.shape
        obs_emb    = self.obs_proj(observations.view(n_agents, t, -1).float())
        action_emb = self.action_emb(actions+1)  # shift by one due to padding idx

        if not self.use_agent_embedding:
            x_t = torch.cat((obs_emb, action_emb), -1)
        else:
            agent_emb = self.agent_emb(
                torch.cat([torch.arange(0, n_agents, 1).view(-1, 1)] * t, 1)
            )
            x_t = torch.cat((obs_emb, agent_emb, action_emb), -1)

        mixed_x_t   = self.mix(x_t)
        output_p, _ = self.gru_actor(input=mixed_x_t,  hx=hidden_actor.swapaxes(1, 0))
        output_c, _ = self.gru_critic(input=mixed_x_t, hx=hidden_critic.swapaxes(1, 0))

        logits = self.action_head(output_p)
        critic = self.critic_head(output_c).squeeze(-1)
        return dict(logits=logits, critic=critic, hidden_actor=output_p, hidden_critic=output_c)


class RecurrentACL2(RecurrentAC):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.action_head = nn.Sequential(
            nn.Linear(self.hidden_size_actor, self.hidden_size_actor),
            nn.Tanh(),
            NormalizedLinear(self.hidden_size_actor, self.n_actions, trainable_magnitude=True)
        )


class NormalizedLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int,
                 device=None, dtype=None, trainable_magnitude=False):
        super(NormalizedLinear, self).__init__(in_features, out_features, False, device, dtype)
        self.d_sqrt = in_features**0.5
        self.trainable_magnitude = trainable_magnitude
        self.scale = nn.Parameter(torch.tensor([1.]), requires_grad=trainable_magnitude)

    def forward(self, input):
        normalized_input = F.normalize(input, dim=-1, p=2, eps=1e-5)
        normalized_weight = F.normalize(self.weight, dim=-1, p=2, eps=1e-5)
        return F.linear(normalized_input, normalized_weight) * self.d_sqrt * self.scale


class L2Norm(nn.Module):
    def __init__(self, in_features, trainable_magnitude=False):
        super(L2Norm, self).__init__()
        self.d_sqrt = in_features**0.5
        self.scale = nn.Parameter(torch.tensor([1.]), requires_grad=trainable_magnitude)

    def forward(self, x):
        return F.normalize(x, dim=-1, p=2, eps=1e-5) * self.d_sqrt * self.scale