import re
import torch
import numpy as np
import yaml
from pathlib import Path
from salina import instantiate_class
from salina import TAgent
from salina.agents.gyma import (
    AutoResetGymAgent,
    _torch_type,
    _format_frame,
    _torch_cat_dict
)


def load_yaml_file(path: Path):
    with path.open() as stream:
        cfg = yaml.load(stream, Loader=yaml.FullLoader)
    return cfg


def add_env_props(cfg):
    env = instantiate_class(cfg['env'].copy())
    cfg['agent'].update(dict(observation_size=env.observation_space.shape,
                             n_actions=env.action_space.n))




AGENT_PREFIX = 'agent#'
REWARD       =  'reward'
CUMU_REWARD  = 'cumulated_reward'
OBS          = 'env_obs'
SEP          = '_'
ACTION       = 'action'


def access_str(agent_i, name, prefix=''):
    return f'{prefix}{AGENT_PREFIX}{agent_i}{SEP}{name}'


class AutoResetGymMultiAgent(AutoResetGymAgent):
    def __init__(self, *args, **kwargs):
        super(AutoResetGymMultiAgent, self).__init__(*args, **kwargs)

    def per_agent_values(self, name, values):
        return {access_str(agent_i, name): value
                for agent_i, value in zip(range(self.n_agents), values)}

    def _initialize_envs(self, n):
        super()._initialize_envs(n)
        n_agents_list = [self.envs[i].unwrapped.n_agents for i in range(n)]
        assert all(n_agents == n_agents_list[0] for n_agents in n_agents_list), \
            'All envs must have the same number of agents.'
        self.n_agents = n_agents_list[0]

    def _reset(self, k, save_render):
        ret = super()._reset(k, save_render)
        obs = ret['env_obs'].squeeze()
        self.cumulated_reward[k] = [0.0]*self.n_agents
        obs      = self.per_agent_values(OBS,  [_format_frame(obs[i]) for i in range(self.n_agents)])
        cumu_rew = self.per_agent_values(CUMU_REWARD, torch.zeros(self.n_agents, 1).float().unbind())
        rewards  = self.per_agent_values(REWARD,      torch.zeros(self.n_agents, 1).float().unbind())
        ret.update(cumu_rew)
        ret.update(rewards)
        ret.update(obs)
        for remove in ['env_obs', 'cumulated_reward', 'reward']:
            del ret[remove]
        return ret

    def _step(self, k, action, save_render):
        self.timestep[k] += 1
        env = self.envs[k]
        if len(action.size()) == 0:
            action = action.item()
            assert isinstance(action, int)
        else:
            action = np.array(action.tolist())
        o, r, d, _ = env.step(action)
        self.cumulated_reward[k] = [x+y for x, y in zip(r, self.cumulated_reward[k])]
        observation = self.per_agent_values(OBS, [_format_frame(o[i]) for i in range(self.n_agents)])
        if d:
            self.is_running[k] = False
        if save_render:
            image = env.render(mode="image").unsqueeze(0)
            observation["rendering"] = image
        rewards           = self.per_agent_values(REWARD, torch.tensor(r).float().view(-1, 1).unbind())
        cumulated_rewards = self.per_agent_values(CUMU_REWARD, torch.tensor(self.cumulated_reward[k]).float().view(-1, 1).unbind())
        ret = {
            **observation,
            **rewards,
            **cumulated_rewards,
            "done": torch.tensor([d]),
            "initial_state": torch.tensor([False]),
            "timestep": torch.tensor([self.timestep[k]])
        }
        return _torch_type(ret)


class CombineActionsAgent(TAgent):
    def __init__(self):
        super().__init__()
        self.pattern = fr'^{AGENT_PREFIX}\d{SEP}{ACTION}$'

    def forward(self, t, **kwargs):
        keys = list(self.workspace.keys())
        action_keys = sorted([k for k in keys if bool(re.match(self.pattern, k))])
        actions = torch.cat([self.get((k, t)) for k in action_keys], 0)
        actions = actions if len(action_keys) <= 1 else actions.unsqueeze(0)
        self.set((f'action', t), actions)
