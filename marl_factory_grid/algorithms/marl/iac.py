import torch
from marl_factory_grid.algorithms.marl.base_ac import BaseActorCritic, nms
from marl_factory_grid.algorithms.utils import instantiate_class
from pathlib import Path
from natsort import natsorted
from marl_factory_grid.algorithms.marl.memory import MARLActorCriticMemory


class LoopIAC(BaseActorCritic):

    def __init__(self, cfg):
        super(LoopIAC, self).__init__(cfg)

    def setup(self):
        self.net = [
            instantiate_class(self.cfg[nms.AGENT]) for _ in range(self.n_agents)
        ]
        self.optimizer = [
            torch.optim.RMSprop(self.net[ag_i].parameters(), lr=3e-4, eps=1e-5) for ag_i in range(self.n_agents)
        ]

    def load_state_dict(self, path: Path):
        paths = natsorted(list(path.glob('*.pt')))
        for path, net in zip(paths, self.net):
            net.load_state_dict(torch.load(path))

    @staticmethod
    def merge_dicts(ds):  # todo could be recursive for more than 1 hierarchy
        d = {}
        for k in ds[0].keys():
            d[k] = [d[k] for d in ds]
        return d

    def init_hidden(self):
        ha  = [net.init_hidden_actor()  for net in self.net]
        hc  = [net.init_hidden_critic() for net in self.net]
        return dict(hidden_actor=ha, hidden_critic=hc)

    def forward(self, observations, actions, hidden_actor, hidden_critic):
        outputs = [
            net(
                self._as_torch(observations[ag_i]).unsqueeze(0).unsqueeze(0),  # agent x time
                self._as_torch(actions[ag_i]).unsqueeze(0),
                hidden_actor[ag_i],
                hidden_critic[ag_i]
                ) for ag_i, net in enumerate(self.net)
        ]
        return self.merge_dicts(outputs)

    def learn(self, tms: MARLActorCriticMemory, **kwargs):
        for ag_i in range(self.n_agents):
            tm, net = tms(ag_i), self.net[ag_i]
            loss = self.actor_critic(tm, net, **self.cfg[nms.ALGORITHM], **kwargs)
            self.optimizer[ag_i].zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 0.5)
            self.optimizer[ag_i].step()