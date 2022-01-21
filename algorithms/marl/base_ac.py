import torch
from typing import Union, List
import numpy as np
from torch.distributions import Categorical
from algorithms.marl.memory import MARLActorCriticMemory
from algorithms.utils import add_env_props, instantiate_class
from pathlib import Path
import pandas as pd
from collections import deque
ListOrTensor = Union[List, torch.Tensor]


class BaseActorCritic:
    def __init__(self, cfg):
        add_env_props(cfg)
        self.__training = True
        self.cfg = cfg
        self.n_agents = cfg['env']['n_agents']
        self.setup()

    def setup(self):
        self.net = instantiate_class(self.cfg['agent'])
        self.optimizer = torch.optim.RMSprop(self.net.parameters(), lr=3e-4, eps=1e-5)

    @classmethod
    def _as_torch(cls, x):
        if isinstance(x, np.ndarray):
            return torch.from_numpy(x)
        elif isinstance(x, List):
            return torch.tensor(x)
        elif isinstance(x, (int, float)):
            return torch.tensor([x])
        return x

    def train(self):
        self.__training = False
        networks = [self.net] if not isinstance(self.net, List) else self.net
        for net in networks:
            net.train()

    def eval(self):
        self.__training = False
        networks = [self.net] if not isinstance(self.net, List) else self.net
        for net in networks:
            net.eval()

    def load_state_dict(self, path: Path):
        pass

    def get_actions(self, out) -> ListOrTensor:
        actions = [Categorical(logits=logits).sample().item() for logits in out['logits']]
        return actions

    def init_hidden(self) -> dict[ListOrTensor]:
        pass

    def forward(self,
                observations:  ListOrTensor,
                actions:       ListOrTensor,
                hidden_actor:  ListOrTensor,
                hidden_critic: ListOrTensor
                ):
        pass


    @torch.no_grad()
    def train_loop(self, checkpointer=None):
        env = instantiate_class(self.cfg['env'])
        n_steps, max_steps = [self.cfg['algorithm'][k] for k in ['n_steps', 'max_steps']]
        global_steps = 0
        reward_queue = deque(maxlen=2000)
        while global_steps < max_steps:
            tm = MARLActorCriticMemory(self.n_agents)
            obs = env.reset()
            last_hiddens        = self.init_hidden()
            last_action, reward = [-1] * self.n_agents, [0.] * self.n_agents
            done, rew_log       = [False]    * self.n_agents, 0
            tm.add(action=last_action, **last_hiddens)

            while not all(done):

                out = self.forward(obs, last_action, **last_hiddens)
                action = self.get_actions(out)
                next_obs, reward, done, info = env.step(action)
                next_obs = next_obs
                if isinstance(done, bool): done = [done] * self.n_agents

                tm.add(observation=obs, action=action, reward=reward, done=done)
                obs = next_obs
                last_action = action
                last_hiddens = dict(hidden_actor=out.get('hidden_actor', None),
                                    hidden_critic=out.get('hidden_critic', None)
                                    )

                if len(tm) >= n_steps or all(done):
                    tm.add(observation=next_obs)
                    if self.__training:
                        with torch.inference_mode(False):
                            self.learn(tm)
                    tm.reset()
                    tm.add(action=last_action, **last_hiddens)
                global_steps += 1
                rew_log += sum(reward)
                reward_queue.extend(reward)

                if checkpointer is not None:
                    checkpointer.step([
                        (f'agent#{i}', agent)
                        for i, agent in enumerate([self.net] if not isinstance(self.net, List) else self.net)
                    ])

                if global_steps >= max_steps: break
            print(f'reward at step: {global_steps} = {rew_log}')

    @torch.inference_mode(True)
    def eval_loop(self, n_episodes, render=False):
        env = instantiate_class(self.cfg['env'])
        episode, results = 0, []
        while episode < n_episodes:
            obs = env.reset()
            last_hiddens           = self.init_hidden()
            last_action, reward    = [-1] * self.n_agents, [0.] * self.n_agents
            done, rew_log, eps_rew = [False] * self.n_agents, 0, torch.zeros(self.n_agents)
            while not all(done):
                if render: env.render()

                out    = self.forward(obs, last_action, **last_hiddens)
                action = self.get_actions(out)
                next_obs, reward, done, info = env.step(action)

                if isinstance(done, bool): done = [done] * obs.shape[0]
                obs = next_obs
                last_action = action
                last_hiddens = dict(hidden_actor=out.get('hidden_actor',   None),
                                    hidden_critic=out.get('hidden_critic', None)
                                    )
                eps_rew += torch.tensor(reward)
            results.append(eps_rew.tolist() + [sum(eps_rew).item()] + [episode])
            episode += 1
        agent_columns = [f'agent#{i}' for i in range(self.cfg['env']['n_agents'])]
        results = pd.DataFrame(results, columns=agent_columns + ['sum', 'episode'])
        results = pd.melt(results, id_vars=['episode'], value_vars=agent_columns + ['sum'], value_name='reward', var_name='agent')
        return results

    @staticmethod
    def compute_advantages(critic, reward, done, gamma):
        return (reward + gamma * (1.0 - done) * critic[:, 1:].detach()) - critic[:, :-1]

    def actor_critic(self, tm, network, gamma, entropy_coef, vf_coef, **kwargs):
        obs, actions, done, reward = tm.observation, tm.action, tm.done, tm.reward

        out = network(obs, actions, tm.hidden_actor, tm.hidden_critic)
        logits = out['logits'][:, :-1]  # last one only needed for v_{t+1}
        critic = out['critic']

        entropy_loss = Categorical(logits=logits).entropy().mean(-1)
        advantages = self.compute_advantages(critic, reward, done, gamma)
        value_loss = advantages.pow(2).mean(-1)  # n_agent

        # policy loss
        log_ap = torch.log_softmax(logits, -1)
        log_ap = torch.gather(log_ap, dim=-1, index=actions[:, 1:].unsqueeze(-1)).squeeze()
        a2c_loss = -(advantages.detach() * log_ap).mean(-1)
        # weighted loss
        loss = a2c_loss + vf_coef*value_loss - entropy_coef * entropy_loss

        return loss.mean()

    def learn(self, tm: MARLActorCriticMemory, **kwargs):
        loss = self.actor_critic(tm, self.net, **self.cfg['algorithm'], **kwargs)
        # remove next_obs, will be added in next iter
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), 0.5)
        self.optimizer.step()

