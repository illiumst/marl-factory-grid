import re
import torch
import yaml
from pathlib import Path
from salina import instantiate_class
from salina import TAgent


def load_yaml_file(path: Path):
    with path.open() as stream:
        cfg = yaml.load(stream, Loader=yaml.FullLoader)
    return cfg


def add_env_props(cfg):
    env = instantiate_class(cfg['env'].copy())
    cfg['agent'].update(dict(observation_size=env.observation_space.shape,
                             n_actions=env.action_space.n))


class CombineActionsAgent(TAgent):
    def __init__(self, pattern=r'^agent\d_action$'):
        super().__init__()
        self.pattern = pattern

    def forward(self, t, **kwargs):
        keys = list(self.workspace.keys())
        action_keys = sorted([k for k in keys if bool(re.match(self.pattern, k))])
        actions = torch.cat([self.get((k, t)) for k in action_keys], 0)
        self.set((f'action', t), actions.unsqueeze(0))
