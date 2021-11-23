from salina.agents.gyma import AutoResetGymAgent
from salina.agents import Agents, TemporalAgent
from salina.rl.functional import _index, gae
import torch
import torch.nn as nn
from torch.distributions import Categorical
from salina import TAgent, Workspace, get_arguments, get_class, instantiate_class
from pathlib import Path
import numpy as np
from tqdm import tqdm
import time
from algorithms.utils import (
    add_env_props,
    load_yaml_file,
    CombineActionsAgent,
    AutoResetGymMultiAgent,
    access_str,
    AGENT_PREFIX, REWARD, CUMU_REWARD, OBS, SEP
)


class A2CAgent(TAgent):
    def __init__(self, observation_size, hidden_size, n_actions, agent_id):
        super().__init__()
        observation_size = np.prod(observation_size)
        print(observation_size)
        self.agent_id = agent_id
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(observation_size, hidden_size),
            nn.ELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ELU()
        )
        self.action_head = nn.Linear(hidden_size, n_actions)
        self.critic_head = nn.Linear(hidden_size, 1)

    def get_obs(self, t):
        observation = self.get((f'env/{access_str(self.agent_id, OBS)}', t))
        return observation

    def forward(self, t, stochastic, **kwargs):
        observation = self.get_obs(t)
        features = self.model(observation)
        scores = self.action_head(features)
        probs = torch.softmax(scores, dim=-1)
        critic = self.critic_head(features).squeeze(-1)
        if stochastic:
            action = torch.distributions.Categorical(probs).sample()
        else:
            action = probs.argmax(1)
        self.set((f'{access_str(self.agent_id, "action")}', t), action)
        self.set((f'{access_str(self.agent_id, "action_probs")}', t), probs)
        self.set((f'{access_str(self.agent_id, "critic")}', t), critic)


if __name__ == '__main__':
    # Setup workspace
    uid = time.time()
    workspace = Workspace()
    n_agents = 2

    # load config
    cfg = load_yaml_file(Path(__file__).parent / 'sat_mad.yaml')
    add_env_props(cfg)
    cfg['env'].update({'n_agents': n_agents})

    # instantiate agent and env
    env_agent = AutoResetGymMultiAgent(
        get_class(cfg['env']),
        get_arguments(cfg['env']),
        n_envs=1
    )

    a2c_agents = [instantiate_class({**cfg['agent'],
                                     'agent_id': agent_id})
                  for agent_id in range(n_agents)]

    # combine agents
    acquisition_agent = TemporalAgent(Agents(env_agent, *a2c_agents, CombineActionsAgent()))
    acquisition_agent.seed(69)

    # optimizers & other parameters
    cfg_optim = cfg['algorithm']['optimizer']
    optimizers = [get_class(cfg_optim)(a2c_agent.parameters(), **get_arguments(cfg_optim))
                  for a2c_agent in a2c_agents]
    n_timesteps = cfg['algorithm']['n_timesteps']

    # Decision making loop
    best = -float('inf')
    with tqdm(range(int(cfg['algorithm']['max_epochs'] / n_timesteps))) as pbar:
        for epoch in pbar:
            workspace.zero_grad()
            if epoch > 0:
                workspace.copy_n_last_steps(1)
                acquisition_agent(workspace, t=1, n_steps=n_timesteps-1, stochastic=True)
            else:
                acquisition_agent(workspace, t=0, n_steps=n_timesteps,  stochastic=True)

            for agent_id in range(n_agents):
                critic, done, action_probs, reward, action = workspace[
                    access_str(agent_id, 'critic'),
                    "env/done",
                    access_str(agent_id, 'action_probs'),
                    access_str(agent_id, 'reward', 'env/'),
                    access_str(agent_id, 'action')
                ]
                td = gae(critic, reward, done, 0.98, 0.25)
                td_error = td ** 2
                critic_loss = td_error.mean()
                entropy_loss = Categorical(action_probs).entropy().mean()
                action_logp = _index(action_probs, action).log()
                a2c_loss = action_logp[:-1] * td.detach()
                a2c_loss = a2c_loss.mean()
                loss = (
                    -0.001 * entropy_loss
                    + 1.0 * critic_loss
                    - 0.1 * a2c_loss
                )
                optimizer = optimizers[agent_id]
                optimizer.zero_grad()
                loss.backward()
                #torch.nn.utils.clip_grad_norm_(a2c_agents[agent_id].parameters(), .5)
                optimizer.step()

                # Compute the cumulated reward on final_state
                rews = ''
                for agent_i in range(n_agents):
                    creward = workspace['env/'+access_str(agent_i, CUMU_REWARD)]
                    creward = creward[done]
                    if creward.size()[0] > 0:
                        rews += f'{AGENT_PREFIX}{agent_i}: {creward.mean().item():.2f}  |  '
                        """if cum_r > best:
                            torch.save(a2c_agent.state_dict(), Path(__file__).parent / f'agent_{uid}.pt')
                            best = cum_r"""
                        pbar.set_description(rews, refresh=True)

