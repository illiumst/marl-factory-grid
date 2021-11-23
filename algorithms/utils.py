import re
import torch
import numpy as np
import yaml
from pathlib import Path
from salina import instantiate_class
from salina import TAgent
from salina.agents.gyma import AutoResetGymAgent, _torch_type, _format_frame


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
        actions = actions if len(action_keys) <= 1 else actions.unsqueeze(0)
        self.set((f'action', t), actions)


class AutoResetGymMultiAgent(AutoResetGymAgent):
    AGENT_PREFIX = 'agent#'
    REWARD       =  'reward'
    CUMU_REWARD  = 'cumulated_reward'
    SEP          = '_'

    def __init__(self, *args, n_agents, **kwargs):
        super(AutoResetGymMultiAgent, self).__init__(*args, **kwargs)
        self.n_agents = n_agents

    def prefix(self, agent_id, name):
        return f'{self.AGENT_PREFIX}{agent_id}{self.SEP}{name}'

    def _reset(self, k, save_render):
        ret = super()._reset(k, save_render)
        self.cumulated_reward[k] = [0.0]*self.n_agents
        del ret['cumulated_reward']
        cumu_rew = {self.prefix(agent_i, self.CUMU_REWARD): torch.zeros(1).float()
                    for agent_i in range(self.n_agents)}
        rewards  = {self.prefix(agent_i, self.REWARD)     : torch.zeros(1).float()
                    for agent_i in range(self.n_agents)}
        ret.update(cumu_rew)
        ret.update(rewards)
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
        print(o.shape)
        observation = _format_frame(o)
        if isinstance(observation, torch.Tensor):
            print(observation.shape)
            observation = {self.prefix(agent_i, 'env_obs'): observation[agent_i]
                           for agent_i in range(self.n_agents)}
            print(observation)
        else:
            assert isinstance(observation, dict)
        if d:
            self.is_running[k] = False

        if save_render:
            image = env.render(mode="image").unsqueeze(0)
            observation["rendering"] = image
        ret = {
            **observation,
            "done": torch.tensor([d]),
            "initial_state": torch.tensor([False]),
            "reward": torch.tensor(r).float(),
            "timestep": torch.tensor([self.timestep[k]]),
            "cumulated_reward": torch.tensor(self.cumulated_reward[k]).float(),
        }
        return _torch_type(ret)

