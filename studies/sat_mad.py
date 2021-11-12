from environments.factory import make
from salina import Workspace, TAgent
from salina.agents.gyma import AutoResetGymAgent, GymAgent
from salina.agents import Agents, TemporalAgent
from salina.rl.functional import _index
import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm
import torch.optim as optim
from torch.distributions import Categorical


class A2CAgent(TAgent):
    def __init__(self, observation_size, hidden_size, n_actions):
        super().__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(observation_size, hidden_size),
            nn.ELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ELU(),
            nn.Linear(hidden_size, n_actions),
        )
        self.critic_model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(observation_size, hidden_size),
            nn.ELU(),
            spectral_norm(nn.Linear(hidden_size, 1)),
        )

    def forward(self, t, stochastic, **kwargs):
        observation = self.get(("env/env_obs", t))
        scores = self.model(observation)
        probs = torch.softmax(scores, dim=-1)
        critic = self.critic_model(observation).squeeze(-1)
        if stochastic:
            action = torch.distributions.Categorical(probs).sample()
        else:
            action = probs.argmax(1)

        self.set(("action", t), action)
        self.set(("action_probs", t), probs)
        self.set(("critic", t), critic)


if __name__ == '__main__':
    # Setup agents and workspace
    env_agent = AutoResetGymAgent(make, dict(env_str='DirtyFactory-v0'), n_envs=1)
    a2c_agent = A2CAgent(3*4*5*5, 96, 10)
    workspace = Workspace()

    eval_agent = Agents(GymAgent(make, dict(env_str='DirtyFactory-v0'), n_envs=1), a2c_agent)
    for i in range(100):
        eval_agent(workspace, t=i, save_render=True, stochastic=True)

    assert False
    # combine agents
    acquisition_agent = TemporalAgent(Agents(env_agent, a2c_agent))
    acquisition_agent.seed(0)

    # optimizers & other parameters
    optimizer = optim.Adam(a2c_agent.parameters(), lr=1e-3)
    n_timesteps = 10

    # Decision making loop
    for epoch in range(200000):
        workspace.zero_grad()
        if epoch > 0:
            workspace.copy_n_last_steps(1)
            acquisition_agent(workspace, t=1, n_steps=n_timesteps-1, stochastic=True)
        else:
            acquisition_agent(workspace, t=0, n_steps=n_timesteps, stochastic=True)
        #for k in workspace.keys():
        #    print(f'{k} ==> {workspace[k].size()}')
        critic, done, action_probs, reward, action = workspace[
            "critic", "env/done", "action_probs", "env/reward", "action"
        ]

        target = reward[1:] + 0.99 * critic[1:].detach() * (1 - done[1:].float())
        td = target - critic[:-1]
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
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Compute the cumulated reward on final_state
        creward = workspace["env/cumulated_reward"]
        creward = creward[done]
        if creward.size()[0] > 0:
            print(f"Cumulative reward at A2C step #{(1+epoch)*n_timesteps}: {creward.mean().item()}")