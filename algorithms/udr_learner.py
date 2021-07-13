import random
from typing import Union, List
from collections import deque
import numpy as np
import torch
import torch.nn as nn
from algorithms.common import BaseBuffer, Experience, BaseLearner, BaseDQN, mlp_maker
from collections import defaultdict


class UDRLBuffer(BaseBuffer):
    def __init__(self, size):
        super(UDRLBuffer, self).__init__(0)
        self.experience = defaultdict(list)
        self.size = size

    def add(self, experience):
        self.experience[experience.episode].append(experience)
        if len(self.experience) > self.size:
            self.sort_and_prune()

    def select_time_steps(self, episode: List[Experience]):
        T = len(episode)  # max horizon
        t1 = random.randint(0, T - 1)
        t2 = random.randint(t1 + 1, T)
        return t1, t2, T

    def sort_and_prune(self):
        scores = []
        for k, episode_experience in self.experience.items():
            r = sum([e.reward for e in episode_experience])
            scores.append((r, k))
        sorted_scores = sorted(scores, reverse=True)
        return sorted_scores

    def sample(self, batch_size, cer=0):
        random_episode_keys = random.choices(list(self.experience.keys()), k=batch_size)
        lsts = (obs, desired_rewards, horizons, actions) = [], [], [], []
        for ek in random_episode_keys:
            episode = self.experience[ek]
            t1, t2, T = self.select_time_steps(episode)
            t2 = T  # TODO only good for episodic envs
            observation = episode[t1].observation
            desired_reward = sum([experience.reward for experience in episode[t1:t2]])
            horizon = t2 - t1
            action = episode[t1].action
            for lst, val in zip(lsts, [observation, desired_reward, horizon, action]):
                lst.append(val)
        return (torch.stack([torch.from_numpy(o) for o in obs], 0).float(),
                torch.tensor(desired_rewards).view(-1, 1).float(),
                torch.tensor(horizons).view(-1, 1).float(),
                torch.tensor(actions))


class UDRLearner(BaseLearner):
    # Upside Down Reinforcement Learner
    def __init__(self, env, desired_reward, desired_horizon,
                 behavior_fn=None, buffer_size=100, n_warm_up_episodes=8, best_x=20,
                 batch_size=128, lr=1e-3, n_agents=1, train_every=('episode', 4), n_grad_steps=1):
        super(UDRLearner, self).__init__(env, n_agents, train_every, n_grad_steps)
        assert self.n_agents == 1, 'UDRL currently only supports single agent training'
        self.behavior_fn = behavior_fn
        self.buffer_size = buffer_size
        self.n_warm_up_episodes = n_warm_up_episodes
        self.buffer = UDRLBuffer(buffer_size)
        self.batch_size = batch_size
        self.mode = 'train'
        self.best_x = best_x
        self.desired_reward = desired_reward
        self.desired_horizon = desired_horizon
        self.lr = lr
        self.optimizer = torch.optim.AdamW(self.behavior_fn.parameters(), lr=lr)

        self.running_loss = deque(maxlen=self.n_grad_steps*5)

    def sample_exploratory_commands(self):
        top_x = self.buffer.sort_and_prune()[:self.best_x]
        # The exploratory desired horizon dh0 is set to the mean of the lengths of the selected episodes
        new_desired_horizon = np.mean([len(self.buffer.experience[k]) for _, k in top_x])
        # save all top_X cumulative returns in a list
        returns = [r for r, _ in top_x]
        # from these returns calc the mean and std
        mean_returns = np.mean([r for r, _ in top_x])
        std_returns = np.std(returns)
        # sample desired reward from a uniform distribution given the mean and the std
        new_desired_reward = np.random.uniform(mean_returns, mean_returns + std_returns)
        self.exploratory_commands = (new_desired_reward, new_desired_horizon)
        return torch.tensor([[new_desired_reward]]).float(), torch.tensor([[new_desired_horizon]]).float()

    def on_new_experience(self, experience):
        self.buffer.add(experience)
        self.desired_reward = self.desired_reward - torch.tensor(experience.reward).float().view(1, 1)

    def on_step_end(self, n_steps):
        one = torch.tensor([1.]).float().view(1, 1)
        self.desired_horizon -= one
        self.desired_horizon = self.desired_horizon if self.desired_horizon >= 1. else one

    def on_episode_end(self, n_steps):
        self.desired_reward, self.desired_horizon = self.sample_exploratory_commands()

    def get_action(self, obs) -> Union[int, np.ndarray]:
        o = torch.from_numpy(obs).unsqueeze(0) if self.n_agents <= 1 else torch.from_numpy(obs)
        bf_out = self.behavior_fn(o.float(), self.desired_reward, self.desired_horizon)
        dist = torch.distributions.Categorical(bf_out)
        sample = dist.sample()
        return [sample.item()]#[self.env.action_space.sample()]

    def _backprop_loss(self, loss):
        # log loss
        self.running_loss.append(loss.item())
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        #torch.nn.utils.clip_grad_norm_(self.behavior_fn.parameters(), 10)
        self.optimizer.step()

    def train(self):
        if len(self.buffer) < self.n_warm_up_episodes: return
        for _ in range(self.n_grad_steps):
            experience = self.buffer.sample(self.batch_size)
            bf_out = self.behavior_fn(*experience[:3])
            labels = experience[-1]
            #print(labels.shape)
            loss = nn.CrossEntropyLoss()(bf_out, labels.squeeze())
            mean_entropy = torch.distributions.Categorical(bf_out).entropy().mean()
            self._backprop_loss(loss - 0.03*mean_entropy)
        print(f'Running loss: {np.mean(list(self.running_loss)):.3f}\tRunning reward: {np.mean(self.running_reward):.2f}'
              f'\td_r: {self.desired_reward.item():.2f}\ttd_h: {self.desired_horizon.item()}')


class BF(BaseDQN):
    def __init__(self, dims=[5*5*3, 64]):
        super(BF, self).__init__(dims)
        self.net = mlp_maker(dims, activation_last='identity')
        self.command_net = mlp_maker([2, 64], activation_last='sigmoid')
        self.common_branch = mlp_maker([64, 64, 64, 9])


    def forward(self, observation, desired_reward, horizon):
        command = torch.cat((desired_reward*(0.02), horizon*(0.01)), dim=-1)
        obs_out = self.net(torch.flatten(observation, start_dim=1))
        command_out = self.command_net(command)
        combined = obs_out*command_out
        out = self.common_branch(combined)
        return torch.softmax(out, -1)


if __name__ == '__main__':
    from environments.factory.simple_factory import SimpleFactory, DirtProperties, MovementProperties
    from algorithms.common import BaseDDQN
    from algorithms.vdn_learner import VDNLearner

    N_AGENTS = 1

    dirt_props = DirtProperties(clean_amount=3, gain_amount=0.2, max_global_amount=30,
                                max_local_amount=5, spawn_frequency=1, max_spawn_ratio=0.05)
    move_props = MovementProperties(allow_diagonal_movement=True,
                                    allow_square_movement=True,
                                    allow_no_op=False)
    env = SimpleFactory(dirt_properties=dirt_props, movement_properties=move_props, n_agents=N_AGENTS, pomdp_radius=2,
                        max_steps=400, omit_agent_slice_in_obs=False, combin_agent_slices_in_obs=True)

    bf = BF()
    desired_reward = torch.tensor([200.]).view(1, 1).float()
    desired_horizon = torch.tensor([400.]).view(1, 1).float()
    learner = UDRLearner(env, behavior_fn=bf,
                         train_every=('episode', 2),
                         buffer_size=40,
                         best_x=10,
                         lr=1e-3,
                         batch_size=64,
                         n_warm_up_episodes=12,
                         n_grad_steps=4,
                         desired_reward=desired_reward,
                         desired_horizon=desired_horizon)
    #learner.save(Path(__file__).parent / 'test' / 'testexperiment1337')
    learner.learn(500000)
