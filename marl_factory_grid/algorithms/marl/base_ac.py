import torch
from typing import Union, List, Dict
import numpy as np
from torch.distributions import Categorical
from marl_factory_grid.algorithms.marl.memory import MARLActorCriticMemory
from marl_factory_grid.algorithms.utils import add_env_props, instantiate_class
from pathlib import Path
import pandas as pd
from collections import deque


class Names:
    REWARD          = 'reward'
    DONE            = 'done'
    ACTION          = 'action'
    OBSERVATION     = 'observation'
    LOGITS          = 'logits'
    HIDDEN_ACTOR    = 'hidden_actor'
    HIDDEN_CRITIC   = 'hidden_critic'
    AGENT           = 'agent'
    ENV             = 'environment'
    N_AGENTS        = 'n_agents'
    ALGORITHM       = 'algorithm'
    MAX_STEPS       = 'max_steps'
    N_STEPS         = 'n_steps'
    BUFFER_SIZE     = 'buffer_size'
    CRITIC          = 'critic'
    BATCH_SIZE      = 'bnatch_size'
    N_ACTIONS       = 'n_actions'

nms = Names
ListOrTensor = Union[List, torch.Tensor]


class BaseActorCritic:
    def __init__(self, cfg):
        add_env_props(cfg)
        self.__training = True
        self.cfg = cfg
        self.n_agents = cfg[nms.ENV][nms.N_AGENTS]
        self.reset_memory_after_epoch = True
        self.setup()

    def setup(self):
        self.net = instantiate_class(self.cfg[nms.AGENT])
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
        actions = [Categorical(logits=logits).sample().item() for logits in out[nms.LOGITS]]
        return actions

    def init_hidden(self) -> Dict[str, ListOrTensor]:
        pass

    def forward(self,
                observations:  ListOrTensor,
                actions:       ListOrTensor,
                hidden_actor:  ListOrTensor,
                hidden_critic: ListOrTensor
                ) -> Dict[str, ListOrTensor]:
        pass

    @torch.no_grad()
    def train_loop(self, checkpointer=None):
        env = instantiate_class(self.cfg[nms.ENV])
        n_steps, max_steps = [self.cfg[nms.ALGORITHM][k] for k in [nms.N_STEPS, nms.MAX_STEPS]]
        tm = MARLActorCriticMemory(self.n_agents, self.cfg[nms.ALGORITHM].get(nms.BUFFER_SIZE, n_steps))
        global_steps, episode, df_results = 0, 0, []
        reward_queue = deque(maxlen=2000)

        while global_steps < max_steps:
            obs = env.reset()
            last_hiddens        = self.init_hidden()
            last_action, reward = [-1] * self.n_agents, [0.] * self.n_agents
            done, rew_log       = [False] * self.n_agents, 0

            if self.reset_memory_after_epoch:
                tm.reset()

            tm.add(observation=obs, action=last_action,
                   logits=torch.zeros(self.n_agents, 1, self.cfg[nms.AGENT][nms.N_ACTIONS]),
                   values=torch.zeros(self.n_agents, 1), reward=reward, done=done, **last_hiddens)

            while not all(done):
                out = self.forward(obs, last_action, **last_hiddens)
                action = self.get_actions(out)
                next_obs, reward, done, info = env.step(action)
                done = [done] * self.n_agents if isinstance(done, bool) else done

                last_hiddens = dict(hidden_actor =out[nms.HIDDEN_ACTOR],
                                    hidden_critic=out[nms.HIDDEN_CRITIC])


                tm.add(observation=obs, action=action, reward=reward, done=done,
                       logits=out.get(nms.LOGITS, None), values=out.get(nms.CRITIC, None),
                       **last_hiddens)

                obs = next_obs
                last_action = action

                if (global_steps+1) % n_steps == 0 or all(done):
                    with torch.inference_mode(False):
                        self.learn(tm)

                global_steps += 1
                rew_log += sum(reward)
                reward_queue.extend(reward)

                if checkpointer is not None:
                    checkpointer.step([
                        (f'agent#{i}', agent)
                        for i, agent in enumerate([self.net] if not isinstance(self.net, List) else self.net)
                    ])

                if global_steps >= max_steps:
                    break
            print(f'reward at episode: {episode} = {rew_log}')
            episode += 1
            df_results.append([episode, rew_log, *reward])
        df_results = pd.DataFrame(df_results, columns=['steps', 'reward', *[f'agent#{i}' for i in range(self.n_agents)]])
        if checkpointer is not None:
            df_results.to_csv(checkpointer.path / 'results.csv', index=False)
        return df_results

    @torch.inference_mode(True)
    def eval_loop(self, n_episodes, render=False):
        env = instantiate_class(self.cfg[nms.ENV])
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
                last_hiddens = dict(hidden_actor=out.get(nms.HIDDEN_ACTOR,   None),
                                    hidden_critic=out.get(nms.HIDDEN_CRITIC, None)
                                    )
                eps_rew += torch.tensor(reward)
            results.append(eps_rew.tolist() + [sum(eps_rew).item()] + [episode])
            episode += 1
        agent_columns = [f'agent#{i}' for i in range(self.cfg['environment']['n_agents'])]
        results = pd.DataFrame(results, columns=agent_columns + ['sum', 'episode'])
        results = pd.melt(results, id_vars=['episode'], value_vars=agent_columns + ['sum'], value_name='reward', var_name='agent')
        return results

    @staticmethod
    def compute_advantages(critic, reward, done, gamma, gae_coef=0.0):
        tds = (reward + gamma * (1.0 - done) * critic[:, 1:].detach()) - critic[:, :-1]

        if gae_coef <= 0:
            return tds

        gae = torch.zeros_like(tds[:, -1])
        gaes = []
        for t in range(tds.shape[1]-1, -1, -1):
            gae = tds[:, t] + gamma * gae_coef * (1.0 - done[:, t]) * gae
            gaes.insert(0, gae)
        gaes = torch.stack(gaes, dim=1)
        return gaes

    def actor_critic(self, tm, network, gamma, entropy_coef, vf_coef, gae_coef=0.0, **kwargs):
        obs, actions, done, reward = tm.observation, tm.action, tm.done[:, 1:], tm.reward[:, 1:]

        out = network(obs, actions, tm.hidden_actor[:, 0], tm.hidden_critic[:, 0])
        logits = out[nms.LOGITS][:, :-1]  # last one only needed for v_{t+1}
        critic = out[nms.CRITIC]

        entropy_loss = Categorical(logits=logits).entropy().mean(-1)
        advantages = self.compute_advantages(critic, reward, done, gamma, gae_coef)
        value_loss = advantages.pow(2).mean(-1)  # n_agent

        # policy loss
        log_ap = torch.log_softmax(logits, -1)
        log_ap = torch.gather(log_ap, dim=-1, index=actions[:, 1:].unsqueeze(-1)).squeeze()
        a2c_loss = -(advantages.detach() * log_ap).mean(-1)
        # weighted loss
        loss = a2c_loss + vf_coef*value_loss - entropy_coef * entropy_loss
        return loss.mean()

    def learn(self, tm: MARLActorCriticMemory, **kwargs):
        loss = self.actor_critic(tm, self.net, **self.cfg[nms.ALGORITHM], **kwargs)
        # remove next_obs, will be added in next iter
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), 0.5)
        self.optimizer.step()

