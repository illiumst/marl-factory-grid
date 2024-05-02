import copy
import random

from scipy import signal
import matplotlib.pyplot as plt
import torch
from typing import Union, List, Dict
import numpy as np
from torch.distributions import Categorical

from marl_factory_grid.algorithms.marl.base_a2c import PolicyGradient, cumulate_discount
from marl_factory_grid.algorithms.marl.memory import MARLActorCriticMemory
from marl_factory_grid.algorithms.utils import add_env_props, instantiate_class
from pathlib import Path
import pandas as pd
from collections import deque
from stable_baselines3 import PPO

from marl_factory_grid.environment.actions import Noop
from marl_factory_grid.modules import Clean, DoorUse


class Names:
    REWARD          = 'reward'
    DONE            = 'done'
    ACTION          = 'action'
    OBSERVATION     = 'observation'
    LOGITS          = 'logits'
    HIDDEN_ACTOR    = 'hidden_actor'
    HIDDEN_CRITIC   = 'hidden_critic'
    AGENT           = 'agent'
    ENV             = 'env'
    ENV_NAME        = 'env_name'
    N_AGENTS        = 'n_agents'
    ALGORITHM       = 'algorithm'
    MAX_STEPS       = 'max_steps'
    N_STEPS         = 'n_steps'
    BUFFER_SIZE     = 'buffer_size'
    CRITIC          = 'critic'
    BATCH_SIZE      = 'bnatch_size'
    N_ACTIONS       = 'n_actions'
    TRAIN_RENDER    = 'train_render'
    EVAL_RENDER     = 'eval_render'


nms = Names
ListOrTensor = Union[List, torch.Tensor]


class A2C:
    def __init__(self, train_cfg, eval_cfg):
        self.factory = add_env_props(train_cfg)
        self.eval_factory = add_env_props(eval_cfg)
        self.__training = True
        self.cfg = train_cfg
        self.n_agents = train_cfg[nms.AGENT][nms.N_AGENTS]
        self.setup()
        self.reward_development = []

    def setup(self):
        # act_dim=6 for dirt_quadrant
        dirt_piles_positions = [self.factory.state.entities['DirtPiles'][pile_idx].pos for pile_idx in
                                range(len(self.factory.state.entities['DirtPiles']))]
        obs_dim = 2 + 2*len(dirt_piles_positions)
        self.agents = [PolicyGradient(self.factory, agent_id=i, obs_dim=obs_dim) for i in range(self.n_agents)]
        self.doors_exist = "Doors" in self.factory.state.entities.keys()

    @classmethod
    def _as_torch(cls, x):
        if isinstance(x, np.ndarray):
            return torch.from_numpy(x)
        elif isinstance(x, List):
            return torch.tensor(x)
        elif isinstance(x, (int, float)):
            return torch.tensor([x])
        return x

    def get_actions(self, observations) -> ListOrTensor:
        # Given an observation, get actions for both agents
        actions = [agent.step(self._as_torch(observations[ag_i]).view(-1).to(torch.float32)) for ag_i, agent in enumerate(self.agents)]
        return actions

    def execute_policy(self, observations) -> ListOrTensor:
        # Use deterministic policy for inference
        actions = [agent.policy(self._as_torch(observations[ag_i]).view(-1).to(torch.float32)) for ag_i, agent in enumerate(self.agents)]
        return actions

    def transform_observations(self, env):
        """ Assumes that agent has observations -DirtPiles and -Self """
        agent_positions = [env.state.moving_entites[agent_idx].pos for agent_idx in range(self.n_agents)]
        dirt_piles_positions = [env.state.entities['DirtPiles'][pile_idx].pos for pile_idx in range(len(env.state.entities['DirtPiles']))]
        trans_obs = [torch.zeros(2+2*len(dirt_piles_positions)) for _ in range(len(agent_positions))]
        for i, pos in enumerate(agent_positions):
            agent_x, agent_y = pos[0], pos[1]
            trans_obs[i][0] = agent_x
            trans_obs[i][1] = agent_y
            idx = 2
            for pos in dirt_piles_positions:
                trans_obs[i][idx] = pos[0]
                trans_obs[i][idx + 1] = pos[1]
                idx += 2
        return trans_obs

    def get_all_observations(self, env):
        first_trans_obs = self.transform_observations(env)[0]
        valid_agent_positions = env.state.entities.floorlist
        #observations_shape = (max(t[0] for t in valid_agent_positions) + 2, max(t[1] for t in valid_agent_positions) + 2)
        observations = []
        for idx, pos in enumerate(valid_agent_positions):
            obs = copy.deepcopy(first_trans_obs)
            obs[0] = pos[0]
            obs[1] = pos[1]
            observations.append(obs)

        return observations

    def get_dirt_piles_positions(self, env):
        return [env.state.entities['DirtPiles'][pile_idx].pos for pile_idx in range(len(env.state.entities['DirtPiles']))]

    def get_ordered_dirt_piles(self, env):
        ordered_dirt_piles = []
        if self.cfg[nms.ALGORITHM]["pile-order"] in ["fixed", "agents"]:
            ordered_dirt_piles = self.get_dirt_piles_positions(env)
        elif self.cfg[nms.ALGORITHM]["pile-order"] == "random":
            ordered_dirt_piles = self.get_dirt_piles_positions(env)
            random.shuffle(ordered_dirt_piles)
        elif self.cfg[nms.ALGORITHM]["pile-order"] == "none":
            ordered_dirt_piles = None
        else:
            print("Not a valid pile order option.")
            exit()

        return ordered_dirt_piles

    def distribute_indices(self, env):
        indices = []
        n_dirt_piles = len(self.get_dirt_piles_positions(env))
        if n_dirt_piles == 1 or self.cfg[nms.ALGORITHM]["pile-order"] in ["fixed", "random", "none"]:
            indices = [[0] for _ in range(self.n_agents)]
        else:
            base_count = n_dirt_piles // self.n_agents
            remainder = n_dirt_piles % self.n_agents

            start_index = 0
            for i in range(self.n_agents):
                # Add an extra index to the first 'remainder' objects
                end_index = start_index + base_count + (1 if i < remainder else 0)
                indices.append(list(range(start_index, end_index)))
                start_index = end_index

        return indices

    def update_target_pile(self, env, agent_idx, target_pile):
        indices = self.distribute_indices(env)
        if target_pile[agent_idx] + 1 in indices[agent_idx]:
            target_pile[agent_idx] += 1

    def door_is_close(self, env, agent_idx):
        neighbourhood = [y for x in env.state.entities.neighboring_positions(env.state["Agent"][agent_idx].pos)
                        for y in env.state.entities.pos_dict[x] if "Door" in y.name]
        if neighbourhood:
            return neighbourhood[0]

    def use_door_or_move(self, env, obs, cleaned_dirt_piles, target_pile, det=False):
        action = []
        for agent_idx, agent in enumerate(self.agents):
            agent_obs = self._as_torch((obs)[agent_idx]).view(-1).to(torch.float32)
            # If agent already reached its target
            if list(cleaned_dirt_piles.values())[target_pile[agent_idx]]:
                action.append(next(action_i for action_i, a in enumerate(env.state["Agent"][agent_idx].actions) if a.name == "Noop"))
                if not det:
                    # Include agent experience entry manually
                    agent._episode.append((None, None, None, agent.vf(agent_obs)))
            else:
                if door := self.door_is_close(env, agent_idx):
                    if door.is_closed:
                        action.append(next(action_i for action_i, a in enumerate(env.state["Agent"][agent_idx].actions) if a.name == "use_door"))
                        if not det:
                            # Include agent experience entry manually
                            agent._episode.append((None, None, None, agent.vf(agent_obs)))
                    else:
                        if det:
                            action.append(int(agent.pi(agent_obs, det=True)[0]))
                        else:
                            action.append(int(agent.step(agent_obs)))
                else:
                    if det:
                        action.append(int(agent.pi(agent_obs, det=True)[0]))
                    else:
                        action.append(int(agent.step(agent_obs)))
        return action

    def reward_distance(self, env, obs, target_pile, reward):
        agent_positions = [env.state.moving_entites[agent_idx].pos for agent_idx in range(self.n_agents)]
        # Give a negative reward for every step that keeps agent from getting closer to currently selected target pile/ closest pile
        for idx, pos in enumerate(agent_positions):
            last_pos = (int(obs[idx][0]), int(obs[idx][1].item()))
            target_pile_pos = self.get_dirt_piles_positions(env)[target_pile[idx]]
            last_distance = np.abs(target_pile_pos[0] - last_pos[0]) + np.abs(target_pile_pos[1] - last_pos[1])
            new_distance = np.abs(target_pile_pos[0] - pos[0]) + np.abs(target_pile_pos[1] - pos[1])
            if new_distance >= last_distance:
                reward[idx] -= 0.05  # 0.05
        return reward

    def punish_entering_same_field(self, next_obs, passed_fields, reward):
        # Give a high negative reward if agent enters same field twice
        for idx in range(self.n_agents):
            if (next_obs[idx][0], next_obs[idx][1]) in passed_fields[idx]:
                reward[idx] += -0.1
            else:
                passed_fields[idx].append((next_obs[idx][0], next_obs[idx][1]))


    def handle_dirt_quadrant_observation_bugs(self, obs, env):
        try:
            # Check that dirt position and amount are still correct
            assert np.where(obs[0][0] == 0.5)[0][0] == 1 and np.where(obs[0][0] == 0.5)[0][0] == 1
        except:
            print("Missing dirt pile")
            # Manually place dirt on defined position
            obs[0][0][1][1] = 0.5
        try:
            # Check that self still returns a valid agent position on the map
            assert np.where(obs[0][1] == 1)[0][0] and np.where(obs[0][1] == 1)[1][0]
        except:
            # Place agent manually in obs object on last known position
            x, y = env.state.moving_entites[0].pos[0], env.state.moving_entites[0].pos[1]
            obs[0][1][x][y] = 1
            print("Missing agent position")

    def handle_dirt(self, env, cleaned_dirt_piles, ordered_dirt_piles, target_pile, reward, done):
        # Check if agent moved on field with dirt. If that is the case collect dirt automatically
        agent_positions = [env.state.moving_entites[agent_idx].pos for agent_idx in range(self.n_agents)]
        dirt_piles_positions = self.get_dirt_piles_positions(env)
        if any([True for pos in agent_positions if pos in dirt_piles_positions]):
            # Do Noop for agent that does not collect dirt
            """action = [np.array(5), np.array(5)]

            # Execute real step in environment
            for idx, pos in enumerate(agent_positions):
                if pos in cleaned_dirt_piles.keys() and not cleaned_dirt_piles[pos]:
                    action[idx] = np.array(4)
                    # Collect dirt
                    _, next_obs, reward, done, info = env.step(action)
                    cleaned_dirt_piles[pos] = True
                    break"""

            # Only simulate collecting the dirt
            for idx, pos in enumerate(agent_positions):
                if pos in self.get_dirt_piles_positions(env) and not cleaned_dirt_piles[pos]:
                    # print(env.state.entities["Agent"][idx], pos, idx, target_pile, ordered_dirt_piles)
                    # If dirt piles should be cleaned in a specific order
                    if ordered_dirt_piles:
                        if pos == ordered_dirt_piles[target_pile[idx]]:
                            reward[idx] += 1  # 1
                            cleaned_dirt_piles[pos] = True
                            # Set pointer to next dirt pile
                            self.update_target_pile(env, idx, target_pile)
                            break
                    else:
                        reward[idx] += 1  # 1
                        cleaned_dirt_piles[pos] = True
                        break

            if all(cleaned_dirt_piles.values()):
                done = True

        return reward, done

    def handle_finished_episode(self, obs):
        with torch.inference_mode(False):
            for ag_i, agent in enumerate(self.agents):
                # Get states, actions, rewards and values from rollout buffer
                (s, a, R, V) = agent.finish_episode()
                # Calculate discounted return and advantage
                G = cumulate_discount(R, self.cfg[nms.ALGORITHM]["gamma"])
                if self.cfg[nms.ALGORITHM]["advantage"] == "Reinforce":
                    A = G
                elif self.cfg[nms.ALGORITHM]["advantage"] == "Advantage-AC":
                    A = G - V  # Actor-Critic Advantages
                elif self.cfg[nms.ALGORITHM]["advantage"] == "TD-Advantage-AC":
                    with torch.no_grad():
                        A = R + self.cfg[nms.ALGORITHM]["gamma"] * np.append(V[1:], agent.vf(
                            self._as_torch(obs[ag_i]).view(-1).to(
                                torch.float32)).numpy()) - V  # TD Actor-Critic Advantages
                else:
                    print("Not a valid advantage option.")
                    exit()

                rollout = (torch.tensor(x.copy()).to(torch.float32) for x in (s, a, G, A))
                # Update policy and value net of agent with experience from rollout buffer
                agent.train(*rollout)


    @torch.no_grad()
    def train_loop(self):
        env = self.factory
        if self.cfg[nms.ENV][nms.TRAIN_RENDER]:
            env.render()
        n_steps, max_steps = [self.cfg[nms.ALGORITHM][k] for k in [nms.N_STEPS, nms.MAX_STEPS]]
        global_steps, episode = 0, 0
        dirt_piles_positions = self.get_dirt_piles_positions(env)
        used_actions = {i:0 for i in range(len(env.state.entities["Agent"][0]._actions))} # Assume both agents have the same actions

        while global_steps < max_steps:
            print(global_steps)
            obs = env.reset() # !!!!!!!!Commented seems to work better? Only if a fixed spawnpoint is given
            print([env.state.moving_entites[agent_idx].pos for agent_idx in range(self.n_agents)])
            """obs = list(obs.values())"""
            obs = self.transform_observations(env)
            done, rew_log       = [False] * self.n_agents, 0

            cleaned_dirt_piles = {pos: False for pos in dirt_piles_positions}
            ordered_dirt_piles = self.get_ordered_dirt_piles(env)
            target_pile = [partition[0] for partition in self.distribute_indices(env)] # pointer that points to the target pile for each agent. (point to same pile, point to different piles)
            """passed_fields = [[] for _ in range(self.n_agents)]"""

            # Add Clean and Noop actions to agent actions so that they can be executed when the agent comes on a dirpile
            """for i in range(self.n_agents):
                self.factory.state['Agent'][i].actions.extend([Clean(), Noop()])"""

            while not all(done):
                # 0="North", 1="East", 2="South", 3="West", 4="Clean", 5="Noop"
                action = self.use_door_or_move(env, obs, cleaned_dirt_piles, target_pile) if self.doors_exist else self.get_actions(obs)
                used_actions[int(action[0])] += 1
                _, next_obs, reward, done, info = env.step(action)
                if done:
                    print("DoneAtMaxStepsReached:", len(self.agents[0]._episode))
                next_obs = self.transform_observations(env)

                # Add small negative reward if agent has moved away from the target_pile
                reward = self.reward_distance(env, obs, target_pile, reward)

                # Check and handle if agent is on field with dirt
                reward, done = self.handle_dirt(env, cleaned_dirt_piles, ordered_dirt_piles, target_pile, reward, done)

                if n_steps != 0 and (global_steps + 1) % n_steps == 0:
                    print("max_steps reached")
                    done = True

                done = [done] * self.n_agents if isinstance(done, bool) else done
                for ag_i, agent in enumerate(self.agents):
                    # Add agent results into respective rollout buffers
                    agent._episode[-1] = (next_obs[ag_i], action[ag_i], reward[ag_i], agent._episode[-1][-1])


                if self.cfg[nms.ENV][nms.TRAIN_RENDER]:
                    env.render()

                obs = next_obs

                if all(done): self.handle_finished_episode(obs)

                global_steps += 1
                rew_log += sum(reward)

                if global_steps >= max_steps:
                    break

            print(f'reward at episode: {episode} = {rew_log}')
            self.reward_development.append(rew_log)
            episode += 1

        # Create value map
        observations_shape = (max(t[0] for t in env.state.entities.floorlist) + 2, max(t[1] for t in env.state.entities.floorlist) + 2)
        value_maps = [np.zeros(observations_shape) for _ in self.agents]
        likeliest_action = [np.full(observations_shape, np.NaN) for _ in self.agents]
        action_probabilities = [np.zeros((observations_shape[0],observations_shape[1], env.action_space[0].n)) for _ in self.agents]
        for obs in self.get_all_observations(env):
            """obs = self._as_torch(obs).view(-1).to(torch.float32)"""
            for idx, agent in enumerate(self.agents):
                """indices = np.where(obs[1] == 1) # Get agent position on grid (1 indicates the position)
                x, y = indices[0][0], indices[1][0]"""
                x, y = int(obs[0]), int(obs[1])
                try:
                    value_maps[idx][x][y] = agent.vf(obs)
                    probs = agent.pi.distribution(obs).probs
                    likeliest_action[idx][x][y] = torch.argmax(probs) # get the likeliest action at the current agent position
                    action_probabilities[idx][x][y] = probs
                except:
                    pass

        print("=======Value Maps=======")
        for agent_idx, vmap in enumerate(value_maps):
            print(f"Value map of agent {agent_idx}:")
            vmap = self._as_torch(vmap).round(decimals=4)
            max_digits = max(len(str(vmap.max().item())), len(str(vmap.min().item())))
            for idx, row in enumerate(vmap):
                print(' '.join(f" {elem:>{max_digits+1}}" for elem in row.tolist()))
        print("=======Likeliest Action=======")
        for agent_idx, amap in enumerate(likeliest_action):
            print(f"Likeliest action map of agent {agent_idx}:")
            print(amap)
        print("=======Action Probabilities=======")
        for agent_idx, pmap in enumerate(action_probabilities):
            print(f"Action probability map of agent {agent_idx}:")
            for d in range(pmap.shape[0]):
                row = '['
                for r in range(pmap.shape[1]):
                    row += "[" + ', '.join(f"{x:7.4f}" for x in pmap[d, r]) + "]"
                print(row + "]")
        print("Used actions:", used_actions)


    @torch.inference_mode(True)
    def eval_loop(self, n_episodes, render=False):
        env = self.eval_factory
        if self.cfg[nms.ENV][nms.EVAL_RENDER]:
            env.render()
        episode, results = 0, []
        dirt_piles_positions = self.get_dirt_piles_positions(env)

        while episode < n_episodes:
            obs = env.reset()
            """obs = list(obs.values())"""
            obs = self.transform_observations(env)
            done, rew_log, eps_rew = [False] * self.n_agents, 0, torch.zeros(self.n_agents)

            cleaned_dirt_piles = {pos: False for pos in dirt_piles_positions}
            ordered_dirt_piles = self.get_ordered_dirt_piles(env)
            target_pile = [partition[0] for partition in self.distribute_indices(env)]

            # Add Clean and Noop actions to agent actions so that they can be executed when the agent comes on a dirpile
            """for i in range(self.n_agents):
                self.factory.state['Agent'][i].actions.extend([Clean(), Noop()])"""

            while not all(done):
                action = self.use_door_or_move(env, obs, cleaned_dirt_piles, target_pile, det=True) if self.doors_exist else self.execute_policy(obs) # zero exploration
                print(action)
                _, next_obs, reward, done, info = env.step(action)
                if done:
                    print("DoneAtMaxStepsReached:", len(self.agents[0]._episode))
                next_obs = self.transform_observations(env)

                # Add small negative reward if agent has moved away from the target_pile
                reward = self.reward_distance(env, obs, target_pile, reward)

                # Check and handle if agent is on field with dirt
                reward, done = self.handle_dirt(env, cleaned_dirt_piles, ordered_dirt_piles, target_pile, reward, done)

                done = [done] * self.n_agents if isinstance(done, bool) else done

                if self.cfg[nms.ENV][nms.EVAL_RENDER]:
                    env.render()

                obs = next_obs

            episode += 1

    def plot_reward_development(self):
        plt.plot(self.reward_development)
        plt.title('Reward development')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.savefig("/Users/julian/Coding/Projects/PyCharmProjects/EDYS/study_out/two_rooms_one_door_modified_runs/reward_development.png")
        plt.show()

