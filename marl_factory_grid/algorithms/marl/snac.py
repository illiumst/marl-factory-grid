from marl_factory_grid.algorithms.marl.base_ac import BaseActorCritic
from marl_factory_grid.algorithms.marl.base_ac import nms
import torch
from torch.distributions import Categorical
from pathlib import Path


class LoopSNAC(BaseActorCritic):
    def __init__(self, cfg):
        super().__init__(cfg)

    def load_state_dict(self, path: Path):
        path2weights = list(path.glob('*.pt'))
        assert len(path2weights) == 1, f'Expected a single set of weights but got {len(path2weights)}'
        self.net.load_state_dict(torch.load(path2weights[0]))

    def init_hidden(self):
        hidden_actor = self.net.init_hidden_actor()
        hidden_critic = self.net.init_hidden_critic()
        return dict(hidden_actor=torch.cat([hidden_actor]   * self.n_agents,  0),
                    hidden_critic=torch.cat([hidden_critic] * self.n_agents,  0)
                    )

    def get_actions(self, out):
        actions = Categorical(logits=out[nms.LOGITS]).sample().squeeze()
        return actions

    def forward(self, observations, actions, hidden_actor, hidden_critic):
        out = self.net(self._as_torch(observations).unsqueeze(1),
                       self._as_torch(actions).unsqueeze(1),
                       hidden_actor, hidden_critic
                       )
        return out