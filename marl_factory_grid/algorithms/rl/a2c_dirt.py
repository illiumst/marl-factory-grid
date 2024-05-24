import copy
import os
import random
import matplotlib.pyplot as plt
import torch
from typing import Union, List
import numpy as np

from marl_factory_grid.algorithms.rl.base_a2c import PolicyGradient, cumulate_discount
from marl_factory_grid.algorithms.utils import add_env_props
from marl_factory_grid.utils.plotting.plot_single_runs import plot_action_maps


class Names:
    ENV             = 'env'
    ENV_NAME        = 'env_name'
    N_AGENTS        = 'n_agents'
    ALGORITHM       = 'algorithm'
    MAX_STEPS       = 'max_steps'
    N_STEPS         = 'n_steps'
    TRAIN_RENDER    = 'train_render'
    EVAL_RENDER     = 'eval_render'


nms = Names
ListOrTensor = Union[List, torch.Tensor]


class A2C:
    def __init__(self, train_cfg, eval_cfg):
        self.factory = add_env_props(train_cfg)
        self.eval_factory = add_env_props(eval_cfg)
        self.__training = True
        self.train_cfg = train_cfg
        self.eval_cfg = eval_cfg
        self.cfg = train_cfg
        self.n_agents = train_cfg[nms.ENV][nms.N_AGENTS]
        self.setup()
        self.reward_development = []
        self.action_probabilities = {agent_idx:[] for agent_idx in range(self.n_agents)}

    def setup(self):
        dirt_piles_positions = [self.factory.state.entities['DirtPiles'][pile_idx].pos for pile_idx in
                                range(len(self.factory.state.entities['DirtPiles']))]
        if self.cfg[nms.ALGORITHM]["pile-observability"] == "all":
            obs_dim = 2 + 2*len(dirt_piles_positions)
        else:
            obs_dim = 4
        self.obs_dim = obs_dim
        self.act_dim = 4
        # act_dim=4, because we want the agent to only learn a routing problem
        self.agents = [PolicyGradient(self.factory, agent_id=i, obs_dim=obs_dim, act_dim=self.act_dim) for i in range(self.n_agents)]
        if self.cfg[nms.ENV]["save_and_log"]:
            # Create results folder
            runs = os.listdir("../study_out/")
            run_numbers = [int(run[3:]) for run in runs if run[:3] == "run"]
            next_run_number = max(run_numbers)+1 if run_numbers else 0
            self.results_path = f"../study_out/run{next_run_number}"
            os.mkdir(self.results_path)
            # Save settings in results folder
            self.save_configs()

    def set_cfg(self, eval=False):
        if eval:
            self.cfg = self.eval_cfg
        else:
            self.cfg = self.train_cfg

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

    def execute_policy(self, observations, env, cleaned_dirt_piles) -> ListOrTensor:
        # Use deterministic policy for inference
        actions = [agent.policy(self._as_torch(observations[ag_i]).view(-1).to(torch.float32)) for ag_i, agent in enumerate(self.agents)]
        for agent_idx in range(self.n_agents):
            if all(cleaned_dirt_piles[agent_idx].values()):
                actions[agent_idx] = np.array(next(action_i for action_i, a in enumerate(env.state["Agent"][agent_idx].actions) if a.name == "Noop"))
        return actions

    def transform_observations(self, env, ordered_dirt_piles, target_pile):
        """ Assumes that agent has observations -DirtPiles and -Self """
        agent_positions = [env.state.moving_entites[agent_idx].pos for agent_idx in range(self.n_agents)]
        if self.cfg[nms.ALGORITHM]["pile-observability"] == "all":
            trans_obs = [torch.zeros(2+2*len(ordered_dirt_piles[0])) for _ in range(len(agent_positions))]
        else:
            # Only show current target pile
            trans_obs = [torch.zeros(4) for _ in range(len(agent_positions))]
        for i, pos in enumerate(agent_positions):
            agent_x, agent_y = pos[0], pos[1]
            trans_obs[i][0] = agent_x
            trans_obs[i][1] = agent_y
            idx = 2
            if self.cfg[nms.ALGORITHM]["pile-observability"] == "all":
                for pile_pos in ordered_dirt_piles[i]:
                    trans_obs[i][idx] = pile_pos[0]
                    trans_obs[i][idx + 1] = pile_pos[1]
                    idx += 2
            else:
                trans_obs[i][2] = ordered_dirt_piles[i][target_pile[i]][0]
                trans_obs[i][3] = ordered_dirt_piles[i][target_pile[i]][1]
        return trans_obs

    def get_all_observations(self, env):
        dirt_piles_positions = [env.state.entities['DirtPiles'][pile_idx].pos for pile_idx in
                                range(len(env.state.entities['DirtPiles']))]
        if self.cfg[nms.ALGORITHM]["pile-observability"] == "all":
            obs = [torch.zeros(2 + 2 * len(dirt_piles_positions))]
            observations = [[]]
            # Fill in pile positions
            idx = 2
            for pile_pos in dirt_piles_positions:
                obs[0][idx] = pile_pos[0]
                obs[0][idx + 1] = pile_pos[1]
                idx += 2
        else:
            # Have multiple observation layers of the map for each dirt pile one
            obs = [torch.zeros(4) for _ in range(self.n_agents) for _ in dirt_piles_positions]
            observations = [[] for _ in dirt_piles_positions]
            for idx, pile_pos in enumerate(dirt_piles_positions):
                obs[idx][2] = pile_pos[0]
                obs[idx][3] = pile_pos[1]
        valid_agent_positions = env.state.entities.floorlist
        #observations_shape = (max(t[0] for t in valid_agent_positions) + 2, max(t[1] for t in valid_agent_positions) + 2)
        for idx, pos in enumerate(valid_agent_positions):
            for obs_layer in range(len(obs)):
                observation = copy.deepcopy(obs[obs_layer])
                observation[0] = pos[0]
                observation[1] = pos[1]
                observations[obs_layer].append(observation)

        return observations

    def get_dirt_piles_positions(self, env):
        return [env.state.entities['DirtPiles'][pile_idx].pos for pile_idx in range(len(env.state.entities['DirtPiles']))]

    def get_ordered_dirt_piles(self, env, cleaned_dirt_piles, target_pile):
        """ Each agent can have it's individual pile order """
        ordered_dirt_piles = [[] for _ in range(self.n_agents)]
        dirt_pile_positions = self.get_dirt_piles_positions(env)
        agent_positions = [env.state.moving_entites[agent_idx].pos for agent_idx in range(self.n_agents)]
        for agent_idx in range(self.n_agents):
            if self.cfg[nms.ALGORITHM]["pile-order"] in ["fixed", "agents"]:
                ordered_dirt_piles[agent_idx] = dirt_pile_positions
            elif self.cfg[nms.ALGORITHM]["pile-order"] == "random":
                ordered_dirt_piles[agent_idx] = dirt_pile_positions
                random.shuffle(ordered_dirt_piles)
            elif self.cfg[nms.ALGORITHM]["pile-order"] == "none":
                ordered_dirt_piles[agent_idx] = None
            elif self.cfg[nms.ALGORITHM]["pile-order"] in ["smart", "dynamic"]:
                # Calculate distances for remaining unvisited dirt piles
                remaining_target_piles = [pos for pos, value in cleaned_dirt_piles[agent_idx].items() if not value]
                pile_distances = {pos:0 for pos in remaining_target_piles}
                agent_pos = agent_positions[agent_idx]
                for pos in remaining_target_piles:
                    pile_distances[pos] = np.abs(agent_pos[0] - pos[0]) + np.abs(agent_pos[1] - pos[1])

                if self.cfg[nms.ALGORITHM]["pile-order"] == "smart":
                    # Check if there is an agent in line with any of the remaining dirt piles
                    for pile_pos in remaining_target_piles:
                        for other_pos in agent_positions:
                            if other_pos != agent_pos:
                                if agent_pos[0] == other_pos[0] == pile_pos[0] or agent_pos[1] == other_pos[1] == pile_pos[1]:
                                    # Get the line between the agent and the goal
                                    path = self.bresenham(agent_pos[0], agent_pos[1], pile_pos[0], pile_pos[1])

                                    # Check if the entity lies on the path between the agent and the goal
                                    if other_pos in path:
                                        pile_distances[pile_pos] += np.abs(agent_pos[0] - other_pos[0]) + np.abs(agent_pos[1] - other_pos[1])

                sorted_pile_distances = dict(sorted(pile_distances.items(), key=lambda item: item[1]))
                # Insert already visited dirt piles
                ordered_dirt_piles[agent_idx] = [pos for pos in dirt_pile_positions if pos not in remaining_target_piles]
                # Fill up with sorted positions
                for pos in sorted_pile_distances.keys():
                    ordered_dirt_piles[agent_idx].append(pos)

            else:
                print("Not a valid pile order option.")
                exit()

        return ordered_dirt_piles

    def bresenham(self, x0, y0, x1, y1):
        """Bresenham's line algorithm to get the coordinates of a line between two points."""
        dx = np.abs(x1 - x0)
        dy = np.abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy

        coordinates = []
        while True:
            coordinates.append((x0, y0))
            if x0 == x1 and y0 == y1:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy
        return coordinates

    def update_ordered_dirt_piles(self, agent_idx, cleaned_dirt_piles, ordered_dirt_piles, env, target_pile):
        # Only update ordered_dirt_pile for agent that reached its target pile
        updated_ordered_dirt_piles = self.get_ordered_dirt_piles(env, cleaned_dirt_piles, target_pile)
        for i in range(len(ordered_dirt_piles[agent_idx])):
            ordered_dirt_piles[agent_idx][i] = updated_ordered_dirt_piles[agent_idx][i]

    def distribute_indices(self, env):
        indices = []
        n_dirt_piles = len(self.get_dirt_piles_positions(env))
        if n_dirt_piles == 1 or self.cfg[nms.ALGORITHM]["pile-order"] in ["fixed", "random", "none", "dynamic", "smart"]:
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

            # Static form: auxiliary pile, primary pile, auxiliary pile, ...
            # -> Starting with index 0 even piles are auxiliary piles, odd piles are primary piles
            if self.cfg[nms.ALGORITHM]["auxiliary_piles"] and "Doors" in env.state.entities.keys():
                door_positions = [door.pos for door in env.state.entities["Doors"]]
                agent_positions = [env.state.moving_entites[agent_idx].pos for agent_idx in range(self.n_agents)]
                distances = {door_pos:[] for door_pos in door_positions}

                # Calculate distance of every agent to every door
                for door_pos in door_positions:
                    for agent_pos in agent_positions:
                        distances[door_pos].append(np.abs(door_pos[0] - agent_pos[0]) + np.abs(door_pos[1] - agent_pos[1]))

                def duplicate_indices(lst, item):
                    return [i for i, x in enumerate(lst) if x == item]

                # Get agent indices of agents with same distance to door
                affected_agents = {door_pos:{} for door_pos in door_positions}
                for door_pos in distances.keys():
                    dist = distances[door_pos]
                    dist_set = set(dist)
                    for d in dist_set:
                        affected_agents[door_pos][str(d)] = duplicate_indices(dist, d)

                # TODO: Make generic for multiple doors
                updated_indices = []
                if len(affected_agents[door_positions[0]]) == 0:
                    # Remove auxiliary piles for all agents
                    # (In config, we defined every pile with an even numbered index to be an auxiliary pile)
                    updated_indices = [[ele for ele in lst if ele % 2 != 0] for lst in indices]
                else:
                    for distance, agent_indices in affected_agents[door_positions[0]].items():
                        # Pick random agent to keep auxiliary pile and remove it for all others
                        #selected_agent = np.random.choice(agent_indices)
                        selected_agent = 0
                        for agent_idx in agent_indices:
                            if agent_idx == selected_agent:
                                updated_indices.append(indices[agent_idx])
                            else:
                                updated_indices.append([ele for ele in indices[agent_idx] if ele % 2 != 0])

                indices = updated_indices

        return indices

    def update_target_pile(self, env, agent_idx, target_pile, indices):
        if self.cfg[nms.ALGORITHM]["pile-order"] in ["fixed", "random", "none", "dynamic", "smart"]:
            if target_pile[agent_idx] + 1 < len(self.get_dirt_piles_positions(env)):
                target_pile[agent_idx] += 1
            else:
                target_pile[agent_idx] = 0
        else:
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
            if all(cleaned_dirt_piles[agent_idx].values()):
                action.append(next(action_i for action_i, a in enumerate(env.state["Agent"][agent_idx].actions) if a.name == "Noop"))
                if not det:
                    # Include agent experience entry manually
                    agent._episode.append((None, None, None, agent.vf(agent_obs)))
            else:
                if door := self.door_is_close(env, agent_idx):
                    if door.is_closed:
                        action.append(next(action_i for action_i, a in enumerate(env.state["Agent"][agent_idx].actions) if a.name == "use_door"))
                        # Don't include action in agent experience
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

    def get_all_cleaned_dirt_piles(self, dirt_piles_positions, cleaned_dirt_piles):
        meta_cleaned_dirt_piles = {pos: False for pos in dirt_piles_positions}
        for agent_idx in range(self.n_agents):
            for (pos, cleaned) in cleaned_dirt_piles[agent_idx].items():
                if cleaned:
                    meta_cleaned_dirt_piles[pos] = True
        return meta_cleaned_dirt_piles

    def handle_dirt(self, env, cleaned_dirt_piles, ordered_dirt_piles, target_pile, indices, reward, done):
        # Check if agent moved on field with dirt. If that is the case collect dirt automatically
        agent_positions = [env.state.moving_entites[agent_idx].pos for agent_idx in range(self.n_agents)]
        dirt_piles_positions = self.get_dirt_piles_positions(env)
        if any([True for pos in agent_positions if pos in dirt_piles_positions]):
            # Do Noop for agent that does not collect dirt
            """action = [np.array(5), np.array(5)]

            # Execute real step in environment
            for idx, pos in enumerate(agent_positions):
                if pos in cleaned_dirt_piles[idx].keys() and not cleaned_dirt_piles[idx][pos]:
                    action[idx] = np.array(4)
                    # Collect dirt
                    _, next_obs, reward, done, info = env.step(action)
                    cleaned_dirt_piles[idx][pos] = True
                    break"""

            # Only simulate collecting the dirt
            for idx, pos in enumerate(agent_positions):
                if pos in cleaned_dirt_piles[idx].keys() and not cleaned_dirt_piles[idx][pos]:
                    # print(env.state.entities["Agent"][idx], pos, idx, target_pile, ordered_dirt_piles)
                    # If dirt piles should be cleaned in a specific order
                    if ordered_dirt_piles[idx]:
                        if pos == ordered_dirt_piles[idx][target_pile[idx]]:
                            reward[idx] += 50  # 1
                            cleaned_dirt_piles[idx][pos] = True
                            # Set pointer to next dirt pile
                            self.update_target_pile(env, idx, target_pile, indices)
                            self.update_ordered_dirt_piles(idx, cleaned_dirt_piles, ordered_dirt_piles, env, target_pile)
                            if self.cfg[nms.ALGORITHM]["pile_all_done"] == "single":
                                done = True
                                if all(cleaned_dirt_piles[idx].values()):
                                    # Reset cleaned_dirt_piles indicator
                                    for pos in dirt_piles_positions:
                                        cleaned_dirt_piles[idx][pos] = False
                    else:
                        reward[idx] += 50  # 1
                        cleaned_dirt_piles[idx][pos] = True

                    # Indicate that renderer can hide dirt pile
                    dirt_at_position = env.state['DirtPiles'].by_pos(pos)
                    dirt_at_position[0].set_new_amount(0)

            if self.cfg[nms.ALGORITHM]["pile_all_done"] in ["all", "distributed"]:
                if all([all(cleaned_dirt_piles[i].values()) for i in range(self.n_agents)]):
                    done = True
            elif self.cfg[nms.ALGORITHM]["pile_all_done"] == "shared":
                # End episode if both agents together have cleaned all dirt piles
                if all(self.get_all_cleaned_dirt_piles(dirt_piles_positions, cleaned_dirt_piles).values()):
                    done = True

        return reward, done

    def handle_finished_episode(self, obs):
        with torch.inference_mode(False):
            for ag_i, agent in enumerate(self.agents):
                # Get states, actions, rewards and values from rollout buffer
                data = agent.finish_episode()
                # Chunk episode data, such that there will be no memory failure for very long episodes
                chunks = self.split_into_chunks(data)
                for (s, a, R, V) in chunks:
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

    def split_into_chunks(self, data_tuple):
        result = [data_tuple]
        chunk_size = self.cfg[nms.ALGORITHM]["chunk-episode"]
        if chunk_size > 0:
            # Get the maximum length of the lists in the tuple to handle different lengths
            max_length = max(len(lst) for lst in data_tuple)

            # Prepare a list to store the result
            result = []

            # Split each list into chunks and add them to the result
            for i in range(0, max_length, chunk_size):
                # Create a sublist containing the ith chunk from each list
                sublist = [lst[i:i + chunk_size] for lst in data_tuple if i < len(lst)]
                result.append(sublist)

        return result

    def set_agent_spawnpoint(self, env):
        for agent_idx in range(self.n_agents):
            agent_name = list(env.state.agents_conf.keys())[agent_idx]
            current_pos_pointer = env.state.agents_conf[agent_name]["pos_pointer"]
            # Making the reset dependent on the number of spawnpoints and not the number of dirtpiles allows
            # for having multiple subsequent spawnpoints with the same target pile
            if current_pos_pointer == len(env.state.agents_conf[agent_name]['positions']) - 1:
                env.state.agents_conf[agent_name]["pos_pointer"] = 0
            else:
                env.state.agents_conf[agent_name]["pos_pointer"] += 1

    @torch.no_grad()
    def train_loop(self):
        env = self.factory
        n_steps, max_steps = [self.cfg[nms.ALGORITHM][k] for k in [nms.N_STEPS, nms.MAX_STEPS]]
        global_steps, episode = 0, 0
        indices = self.distribute_indices(env)
        dirt_piles_positions = self.get_dirt_piles_positions(env)
        used_actions = {i:0 for i in range(len(env.state.entities["Agent"][0]._actions))} # Assume both agents have the same actions
        target_pile = [partition[0] for partition in indices]  # pointer that points to the target pile for each agent. (point to same pile, point to different piles)
        cleaned_dirt_piles = [{pos: False for pos in dirt_piles_positions} for _ in range(self.n_agents)] # Have own dictionary for each agent

        while global_steps < max_steps:
            print(global_steps)
            obs = env.reset() # !!!!!!!!Commented seems to work better? Only if a fixed spawnpoint is given
            if self.cfg[nms.ENV][nms.TRAIN_RENDER]:
                env.render()
            self.set_agent_spawnpoint(env)
            ordered_dirt_piles = self.get_ordered_dirt_piles(env, cleaned_dirt_piles, target_pile)
            # Reset current target pile at episode begin if all piles have to be cleaned in one episode
            if self.cfg[nms.ALGORITHM]["pile_all_done"] == "all":
                target_pile = [partition[0] for partition in indices]
                cleaned_dirt_piles = [{pos: False for pos in dirt_piles_positions} for _ in range(self.n_agents)]
            """passed_fields = [[] for _ in range(self.n_agents)]"""

            """obs = list(obs.values())"""
            obs = self.transform_observations(env, ordered_dirt_piles, target_pile)
            done, rew_log       = [False] * self.n_agents, 0

            print("Agents spawnpoints:", [env.state.moving_entites[agent_idx].pos for agent_idx in range(self.n_agents)])
            print("Agents target piles:", target_pile)
            print("Agents initial observation:", obs)
            print("Agents cleaned dirt piles:", cleaned_dirt_piles)

            # Add Clean and Noop actions to agent actions so that they can be executed when the agent comes on a dirpile
            """for i in range(self.n_agents):
                self.factory.state['Agent'][i].actions.extend([Clean(), Noop()])"""

            while not all(done):
                # 0="North", 1="East", 2="South", 3="West", 4="Clean", 5="Noop"
                action = self.use_door_or_move(env, obs, cleaned_dirt_piles, target_pile) \
                    if "Doors" in env.state.entities.keys() else self.get_actions(obs)
                used_actions[int(action[0])] += 1
                _, next_obs, reward, done, info = env.step(action)
                if done:
                    print("DoneAtMaxStepsReached:", len(self.agents[0]._episode))
                next_obs = self.transform_observations(env, ordered_dirt_piles, target_pile)

                # Add small negative reward if agent has moved away from the target_pile
                # reward = self.reward_distance(env, obs, target_pile, reward)

                # Check and handle if agent is on field with dirt. This method can change the observation for the next step.
                # If pile_all_done is "single", the episode ends if agents reached its target pile and the new episode begins
                # with the updated observation. The observation that is saved to the rollout buffer, which resulted in reaching
                # the target pile should not be updated before saving. Thus, the self.transform_observations call must happen
                # before this method is called.
                reward, done = self.handle_dirt(env, cleaned_dirt_piles, ordered_dirt_piles, target_pile, indices, reward, done)

                if n_steps != 0 and (global_steps + 1) % n_steps == 0:
                    print("max_steps reached")
                    done = True

                done = [done] * self.n_agents if isinstance(done, bool) else done
                for ag_i, agent in enumerate(self.agents):
                    # For forced actions like door opening, we have to call the step function with this action, but
                    # since we are not allowed to exceed the dimensions range, we can't log the corresponding step info.
                    if action[ag_i] in range(self.act_dim):
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

        self.plot_reward_development()
        if self.cfg[nms.ENV]["save_and_log"]:
            self.create_info_maps(env, used_actions)
            self.save_agent_models()
            plot_action_maps(env, [self], self.results_path)

    @torch.inference_mode(True)
    def eval_loop(self, n_episodes, render=False):
        env = self.eval_factory
        self.set_cfg(eval=True)
        episode, results = 0, []
        dirt_piles_positions = self.get_dirt_piles_positions(env)
        indices = self.distribute_indices(env)
        target_pile = [partition[0] for partition in indices]  # pointer that points to the target pile for each agent. (point to same pile, point to different piles)
        if self.cfg[nms.ALGORITHM]["pile_all_done"] == "distributed":
            cleaned_dirt_piles = [{dirt_piles_positions[idx]: False for idx in indices[i]} for i in range(self.n_agents)]
        else:
            cleaned_dirt_piles = [{pos: False for pos in dirt_piles_positions} for _ in range(self.n_agents)]

        while episode < n_episodes:
            obs = env.reset()
            self.set_agent_spawnpoint(env)
            if self.cfg[nms.ENV][nms.EVAL_RENDER]:
                if self.cfg[nms.ALGORITHM]["auxiliary_piles"]:
                    # Don't render auxiliary piles
                    auxiliary_piles = [pile for idx, pile in enumerate(env.state.entities['DirtPiles']) if idx % 2 == 0]
                    for pile in auxiliary_piles:
                        pile.set_new_amount(0)
                env.render()
                env._renderer.fps = 5
            """obs = list(obs.values())"""
            # Reset current target pile at episode begin if all piles have to be cleaned in one episode
            if self.cfg[nms.ALGORITHM]["pile_all_done"] in ["all", "distributed", "shared"]:
                target_pile = [partition[0] for partition in indices]
                if self.cfg[nms.ALGORITHM]["pile_all_done"] == "distributed":
                    cleaned_dirt_piles = [{dirt_piles_positions[idx]: False for idx in indices[i]} for i in range(self.n_agents)]
                else:
                    cleaned_dirt_piles = [{pos: False for pos in dirt_piles_positions} for _ in range(self.n_agents)]

            ordered_dirt_piles = self.get_ordered_dirt_piles(env, cleaned_dirt_piles, target_pile)

            obs = self.transform_observations(env, ordered_dirt_piles, target_pile)
            done, rew_log, eps_rew = [False] * self.n_agents, 0, torch.zeros(self.n_agents)

            # Add Clean and Noop actions to agent actions so that they can be executed when the agent comes on a dirpile
            """for i in range(self.n_agents):
                self.factory.state['Agent'][i].actions.extend([Clean(), Noop()])"""

            while not all(done):
                action = self.use_door_or_move(env, obs, cleaned_dirt_piles, target_pile, det=True) \
                    if "Doors" in env.state.entities.keys() else self.execute_policy(obs, env, cleaned_dirt_piles) # zero exploration
                _, next_obs, reward, done, info = env.step(action) # Note that this call seems to flip the lists in indices
                if done:
                    print("DoneAtMaxStepsReached:", len(self.agents[0]._episode))

                # Add small negative reward if agent has moved away from the target_pile
                # reward = self.reward_distance(env, obs, target_pile, reward)

                # Check and handle if agent is on field with dirt
                reward, done = self.handle_dirt(env, cleaned_dirt_piles, ordered_dirt_piles, target_pile, indices, reward, done)

                # Get transformed next_obs that might have been updated because of self.handle_dirt.
                # For eval, where pile_all_done is "all", it's mandatory that the potential change of the target pile
                # in the observation, caused by self.handle_dirt, is already considered when the next action is calculated.
                next_obs = self.transform_observations(env, ordered_dirt_piles, target_pile)

                done = [done] * self.n_agents if isinstance(done, bool) else done

                if self.cfg[nms.ENV][nms.EVAL_RENDER]:
                    env.render()

                obs = next_obs

            episode += 1

    def plot_reward_development(self):
        smoothed_data = np.convolve(self.reward_development, np.ones(10) / 10, mode='valid')
        plt.plot(smoothed_data)
        plt.ylim([-10, max(smoothed_data) + 20])
        plt.title('Smoothed Reward Development')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        if self.cfg[nms.ENV]["save_and_log"]:
            plt.savefig(f"{self.results_path}/smoothed_reward_development.png")
        plt.show()

    def save_configs(self):
        with open(f"{self.results_path}/MARL_config.txt", "w") as txt_file:
            txt_file.write(str(self.cfg))
        with open(f"{self.results_path}/train_env_config.txt", "w") as txt_file:
            txt_file.write(str(self.factory.conf))
        with open(f"{self.results_path}/eval_env_config.txt", "w") as txt_file:
            txt_file.write(str(self.eval_factory.conf))

    def save_agent_models(self):
        for idx, agent in enumerate(self.agents):
            agent.pi.save_model_parameters(self.results_path)
            agent.vf.save_model_parameters(self.results_path)

    def load_agents(self, runs_list):
        for idx, run in enumerate(runs_list):
            run_path = f"../study_out/{run}"
            self.agents[idx].pi.load_model_parameters(f"{run_path}/PolicyNet_model_parameters.pth")
            self.agents[idx].vf.load_model_parameters(f"{run_path}/ValueNet_model_parameters.pth")

    def create_info_maps(self, env, used_actions):
        # Create value map
        all_valid_observations = self.get_all_observations(env)
        dirt_piles_positions = self.get_dirt_piles_positions(env)
        with open(f"{self.results_path}/info_maps.txt", "w") as txt_file:
            for obs_layer, pos in enumerate(dirt_piles_positions):
                observations_shape = (
                max(t[0] for t in env.state.entities.floorlist) + 2, max(t[1] for t in env.state.entities.floorlist) + 2)
                value_maps = [np.zeros(observations_shape) for _ in self.agents]
                likeliest_action = [np.full(observations_shape, np.NaN) for _ in self.agents]
                action_probabilities = [np.zeros((observations_shape[0], observations_shape[1], self.act_dim)) for
                                        _ in self.agents]
                for obs in all_valid_observations[obs_layer]:
                    """obs = self._as_torch(obs).view(-1).to(torch.float32)"""
                    for idx, agent in enumerate(self.agents):
                        """indices = np.where(obs[1] == 1) # Get agent position on grid (1 indicates the position)
                        x, y = indices[0][0], indices[1][0]"""
                        x, y = int(obs[0]), int(obs[1])
                        try:
                            value_maps[idx][x][y] = agent.vf(obs)
                            probs = agent.pi.distribution(obs).probs
                            likeliest_action[idx][x][y] = torch.argmax(probs)  # get the likeliest action at the current agent position
                            action_probabilities[idx][x][y] = probs
                        except:
                            pass

                txt_file.write("=======Value Maps=======\n")
                print("=======Value Maps=======")
                for agent_idx, vmap in enumerate(value_maps):
                    txt_file.write(f"Value map of agent {agent_idx} for target pile {pos}:\n")
                    print(f"Value map of agent {agent_idx} for target pile {pos}:")
                    vmap = self._as_torch(vmap).round(decimals=4)
                    max_digits = max(len(str(vmap.max().item())), len(str(vmap.min().item())))
                    for idx, row in enumerate(vmap):
                        txt_file.write(' '.join(f" {elem:>{max_digits + 1}}" for elem in row.tolist()))
                        txt_file.write("\n")
                        print(' '.join(f" {elem:>{max_digits + 1}}" for elem in row.tolist()))
                txt_file.write("\n")
                txt_file.write("=======Likeliest Action=======\n")
                print("=======Likeliest Action=======")
                for agent_idx, amap in enumerate(likeliest_action):
                    txt_file.write(f"Likeliest action map of agent {agent_idx} for target pile {pos}:\n")
                    print(f"Likeliest action map of agent {agent_idx} for target pile {pos}:")
                    txt_file.write(np.array2string(amap))
                    print(amap)
                txt_file.write("\n")
                txt_file.write("=======Action Probabilities=======\n")
                print("=======Action Probabilities=======")
                for agent_idx, pmap in enumerate(action_probabilities):
                    self.action_probabilities[agent_idx].append(pmap)
                    txt_file.write(f"Action probability map of agent {agent_idx} for target pile {pos}:\n")
                    print(f"Action probability map of agent {agent_idx} for target pile {pos}:")
                    for d in range(pmap.shape[0]):
                        row = '['
                        for r in range(pmap.shape[1]):
                            row += "[" + ', '.join(f"{x:7.4f}" for x in pmap[d, r]) + "]"
                        txt_file.write(row + "]")
                        txt_file.write("\n")
                        print(row + "]")
                txt_file.write(f"Used actions: {used_actions}\n")
                print("Used actions:", used_actions)

