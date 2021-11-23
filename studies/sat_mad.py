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
    AutoResetGymMultiAgent
)


class A2CAgent(TAgent):
    def __init__(self, observation_size, hidden_size, n_actions, agent_id=-1, marl=False):
        super().__init__()
        observation_size = np.prod(observation_size)
        self.agent_id = agent_id
        self.marl = marl
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
        observation = self.get(("env/env_obs", t))
        print(observation.shape)
        if self.marl:
            observation = observation[self.agent_id]
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
        agent_str = f'agent{self.agent_id}_'
        self.set((f'{agent_str}action', t), action)
        self.set((f'{agent_str}action_probs', t), probs)
        self.set((f'{agent_str}critic', t), critic)


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
        n_envs=1,
        n_agents=n_agents
    )

    a2c_agents = [instantiate_class({**cfg['agent'],
                                     'agent_id': agent_id,
                                     'marl':     n_agents > 1})
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
                    f"agent{agent_id}_critic", "env/done",
                    f'agent{agent_id}_action_probs', "env/reward",
                    f"agent{agent_id}_action"
                ]
                reward = reward[agent_id]
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
                creward = workspace["env/cumulated_reward"]#[agent_id].unsqueeze(-1)
                print(creward.shape, done.shape)
                creward = creward[done]
                if creward.size()[0] > 0:
                    cum_r = creward.mean().item()
                    if cum_r > best:
                    #    torch.save(a2c_agent.state_dict(), Path(__file__).parent / f'agent_{uid}.pt')
                        best = cum_r
                    pbar.set_description(f"Cum. r: {cum_r:.2f}, Best r. so far: {best:.2f}", refresh=True)

