import os
import torch
from typing import Union, List
import numpy as np
from tqdm import tqdm

from marl_factory_grid.algorithms.rl.base_a2c import PolicyGradient, cumulate_discount
from marl_factory_grid.algorithms.rl.constants import Names
from marl_factory_grid.algorithms.rl.utils import transform_observations, _as_torch, is_door_close, \
    get_dirt_piles_positions, update_target_pile, update_ordered_dirt_piles, get_all_cleaned_dirt_piles, \
    distribute_indices, set_agents_spawnpoints, get_ordered_dirt_piles, handle_finished_episode, save_configs, \
    save_agent_models, get_all_observations, get_agents_positions
from marl_factory_grid.algorithms.utils import add_env_props
from marl_factory_grid.utils.plotting.plot_single_runs import plot_action_maps, plot_reward_development, \
    create_info_maps

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
        self.action_probabilities = {agent_idx: [] for agent_idx in range(self.n_agents)}

    def setup(self):
        """ Initialize agents and create entry for run results according to configuration """
        self.obs_dim = 2 + 2 * len(get_dirt_piles_positions(self.factory)) if self.cfg[nms.ALGORITHM][
                                                                                  nms.PILE_OBSERVABILITY] == nms.ALL else 4
        self.act_dim = 4  # The 4 movement directions
        self.agents = [PolicyGradient(self.factory, agent_id=i, obs_dim=self.obs_dim, act_dim=self.act_dim) for i in
                       range(self.n_agents)]

        if self.cfg[nms.ENV][nms.SAVE_AND_LOG]:
            # Create results folder
            runs = os.listdir("./study_out/")
            run_numbers = [int(run[3:]) for run in runs if run[:3] == "run"]
            next_run_number = max(run_numbers) + 1 if run_numbers else 0
            self.results_path = f"./study_out/run{next_run_number}"
            os.mkdir(self.results_path)
            # Save settings in results folder
            save_configs(self.results_path, self.cfg, self.factory.conf, self.eval_factory.conf)

    def set_cfg(self, eval=False):
        """ Set the mode of the current configuration """
        if eval:
            self.cfg = self.eval_cfg
        else:
            self.cfg = self.train_cfg

    def load_agents(self, runs_list):
        """ Initialize networks with parameters of already trained agents """
        for idx, run in enumerate(runs_list):
            run_path = f"./study_out/{run}"
            self.agents[idx].pi.load_model_parameters(f"{run_path}/PolicyNet_model_parameters.pth")
            self.agents[idx].vf.load_model_parameters(f"{run_path}/ValueNet_model_parameters.pth")

    @torch.no_grad()
    def train_loop(self):
        """ Function for training agents """
        env = self.factory
        n_steps, max_steps = [self.cfg[nms.ALGORITHM][k] for k in [nms.N_STEPS, nms.MAX_STEPS]]
        global_steps, episode = 0, 0
        indices = distribute_indices(env, self.cfg, self.n_agents)
        dirt_piles_positions = get_dirt_piles_positions(env)
        target_pile = [partition[0] for partition in
                       indices]  # list of pointers that point to the current target pile for each agent
        cleaned_dirt_piles = [{pos: False for pos in dirt_piles_positions} for _ in range(self.n_agents)]

        pbar = tqdm(total=max_steps)
        while global_steps < max_steps:
            _ = env.reset()
            if self.cfg[nms.ENV][nms.TRAIN_RENDER]:
                env.render()
            set_agents_spawnpoints(env, self.n_agents)
            ordered_dirt_piles = get_ordered_dirt_piles(env, cleaned_dirt_piles, self.cfg, self.n_agents)
            # Reset current target pile at episode begin if all piles have to be cleaned in one episode
            if self.cfg[nms.ALGORITHM][nms.PILE_ALL_DONE] == nms.ALL:
                target_pile = [partition[0] for partition in indices]
                cleaned_dirt_piles = [{pos: False for pos in dirt_piles_positions} for _ in range(self.n_agents)]

            # Supply each agent with its local observation
            obs = transform_observations(env, ordered_dirt_piles, target_pile, self.cfg, self.n_agents)
            done, rew_log = [False] * self.n_agents, 0

            while not all(done):
                action = self.use_door_or_move(env, obs, cleaned_dirt_piles) \
                    if nms.DOORS in env.state.entities.keys() else self.get_actions(obs)
                _, next_obs, reward, done, info = env.step(action)
                next_obs = transform_observations(env, ordered_dirt_piles, target_pile, self.cfg, self.n_agents)

                # Handle case where agent is on field with dirt
                reward, done = self.handle_dirt(env, cleaned_dirt_piles, ordered_dirt_piles, target_pile, indices,
                                                reward, done)

                if n_steps != 0 and (global_steps + 1) % n_steps == 0: done = True

                done = [done] * self.n_agents if isinstance(done, bool) else done
                for ag_i, agent in enumerate(self.agents):
                    if action[ag_i] in range(self.act_dim):
                        # Add agent results into respective rollout buffers
                        agent._episode[-1] = (next_obs[ag_i], action[ag_i], reward[ag_i], agent._episode[-1][-1])

                # Visualize state update
                if self.cfg[nms.ENV][nms.TRAIN_RENDER]: env.render()

                obs = next_obs

                if all(done): handle_finished_episode(obs, self.agents, self.cfg)

                global_steps += 1
                rew_log += sum(reward)

                if global_steps >= max_steps: break

            self.reward_development.append(rew_log)
            episode += 1
            pbar.update(global_steps - pbar.n)

        pbar.close()
        if self.cfg[nms.ENV][nms.SAVE_AND_LOG]:
            plot_reward_development(self.reward_development, self.results_path)
            create_info_maps(env, get_all_observations(env, self.cfg, self.n_agents),
                             get_dirt_piles_positions(env), self.results_path, self.agents, self.act_dim, self)
            save_agent_models(self.results_path, self.agents)
            plot_action_maps(env, [self], self.results_path)

    @torch.inference_mode(True)
    def eval_loop(self, n_episodes):
        """ Function for performing inference """
        env = self.eval_factory
        self.set_cfg(eval=True)
        episode, results = 0, []
        dirt_piles_positions = get_dirt_piles_positions(env)
        indices = distribute_indices(env, self.cfg, self.n_agents)
        target_pile = [partition[0] for partition in
                       indices]  # list of pointers that point to the current target pile for each agent
        if self.cfg[nms.ALGORITHM][nms.PILE_ALL_DONE] == nms.DISTRIBUTED:
            cleaned_dirt_piles = [{dirt_piles_positions[idx]: False for idx in indices[i]} for i in
                                  range(self.n_agents)]
        else: cleaned_dirt_piles = [{pos: False for pos in dirt_piles_positions} for _ in range(self.n_agents)]

        while episode < n_episodes:
            _ = env.reset()
            set_agents_spawnpoints(env, self.n_agents)
            if self.cfg[nms.ENV][nms.EVAL_RENDER]:
                # Don't render auxiliary piles
                if self.cfg[nms.ALGORITHM][nms.AUXILIARY_PILES]:
                    auxiliary_piles = [pile for idx, pile in enumerate(env.state.entities[nms.DIRT_PILES]) if
                                       idx % 2 == 0]
                    for pile in auxiliary_piles:
                        pile.set_new_amount(0)
                env.render()
                env._renderer.fps = 5  # Slow down agent movement

            # Reset current target pile at episode begin if all piles have to be cleaned in one episode
            if self.cfg[nms.ALGORITHM][nms.PILE_ALL_DONE] in [nms.ALL, nms.DISTRIBUTED, nms.SHARED]:
                target_pile = [partition[0] for partition in indices]
                if self.cfg[nms.ALGORITHM][nms.PILE_ALL_DONE] == nms.DISTRIBUTED:
                    cleaned_dirt_piles = [{dirt_piles_positions[idx]: False for idx in indices[i]} for i in
                                          range(self.n_agents)]
                else: cleaned_dirt_piles = [{pos: False for pos in dirt_piles_positions} for _ in range(self.n_agents)]

            ordered_dirt_piles = get_ordered_dirt_piles(env, cleaned_dirt_piles, self.cfg, self.n_agents)

            # Supply each agent with its local observation
            obs = transform_observations(env, ordered_dirt_piles, target_pile, self.cfg, self.n_agents)
            done, rew_log, eps_rew = [False] * self.n_agents, 0, torch.zeros(self.n_agents)

            while not all(done):
                action = self.use_door_or_move(env, obs, cleaned_dirt_piles, det=True) \
                    if nms.DOORS in env.state.entities.keys() else self.execute_policy(obs, env,
                                                                                       cleaned_dirt_piles)  # zero exploration
                _, next_obs, reward, done, info = env.step(action)

                # Handle case where agent is on field with dirt
                reward, done = self.handle_dirt(env, cleaned_dirt_piles, ordered_dirt_piles, target_pile, indices,
                                                reward, done)

                # Get transformed next_obs that might have been updated because of handle_dirt
                next_obs = transform_observations(env, ordered_dirt_piles, target_pile, self.cfg, self.n_agents)

                done = [done] * self.n_agents if isinstance(done, bool) else done

                if self.cfg[nms.ENV][nms.EVAL_RENDER]: env.render()

                obs = next_obs

            episode += 1

    ########## Helper functions ########

    def get_actions(self, observations) -> ListOrTensor:
        """ Given local observations, get actions for both agents """
        actions = [agent.step(_as_torch(observations[ag_i]).view(-1).to(torch.float32)) for ag_i, agent in
                   enumerate(self.agents)]
        return actions

    def execute_policy(self, observations, env, cleaned_dirt_piles) -> ListOrTensor:
        """ Execute agent policies deterministically for inference """
        actions = [agent.policy(_as_torch(observations[ag_i]).view(-1).to(torch.float32)) for ag_i, agent in
                   enumerate(self.agents)]
        for agent_idx in range(self.n_agents):
            if all(cleaned_dirt_piles[agent_idx].values()):
                actions[agent_idx] = np.array(next(
                    action_i for action_i, a in enumerate(env.state[nms.AGENT][agent_idx].actions) if
                    a.name == nms.NOOP))
        return actions

    def use_door_or_move(self, env, obs, cleaned_dirt_piles, det=False):
        """ Function that handles automatic actions like door opening and forced Noop"""
        action = []
        for agent_idx, agent in enumerate(self.agents):
            agent_obs = _as_torch((obs)[agent_idx]).view(-1).to(torch.float32)
            # Use Noop operation if agent already reached its target. (Only relevant for two-rooms setting)
            if all(cleaned_dirt_piles[agent_idx].values()):
                action.append(next(action_i for action_i, a in enumerate(env.state[nms.AGENT][agent_idx].actions) if
                                   a.name == nms.NOOP))
                if not det:
                    # Include agent experience entry manually
                    agent._episode.append((None, None, None, agent.vf(agent_obs)))
            else:
                if door := is_door_close(env, agent_idx):
                    if door.is_closed:
                        action.append(next(
                            action_i for action_i, a in enumerate(env.state[nms.AGENT][agent_idx].actions) if
                            a.name == nms.USE_DOOR))
                        # Don't include action in agent experience
                    else:
                        if det: action.append(int(agent.pi(agent_obs, det=True)[0]))
                        else: action.append(int(agent.step(agent_obs)))
                else:
                    if det: action.append(int(agent.pi(agent_obs, det=True)[0]))
                    else: action.append(int(agent.step(agent_obs)))
        return action

    def handle_dirt(self, env, cleaned_dirt_piles, ordered_dirt_piles, target_pile, indices, reward, done):
        """ Check if agent moved on field with dirt. If that is the case collect dirt automatically """
        agents_positions = get_agents_positions(env, self.n_agents)
        dirt_piles_positions = get_dirt_piles_positions(env)
        if any([True for pos in agents_positions if pos in dirt_piles_positions]):
            # Only simulate collecting the dirt
            for idx, pos in enumerate(agents_positions):
                if pos in cleaned_dirt_piles[idx].keys() and not cleaned_dirt_piles[idx][pos]:

                    # If dirt piles should be cleaned in a specific order
                    if ordered_dirt_piles[idx]:
                        if pos == ordered_dirt_piles[idx][target_pile[idx]]:
                            reward[idx] += 50
                            cleaned_dirt_piles[idx][pos] = True
                            # Set pointer to next dirt pile
                            update_target_pile(env, idx, target_pile, indices, self.cfg)
                            update_ordered_dirt_piles(idx, cleaned_dirt_piles, ordered_dirt_piles, env,
                                                      self.cfg, self.n_agents)
                            if self.cfg[nms.ALGORITHM][nms.PILE_ALL_DONE] == nms.SINGLE:
                                done = True
                                if all(cleaned_dirt_piles[idx].values()):
                                    # Reset cleaned_dirt_piles indicator
                                    for pos in dirt_piles_positions:
                                        cleaned_dirt_piles[idx][pos] = False
                    else:
                        reward[idx] += 50
                        cleaned_dirt_piles[idx][pos] = True

                    # Indicate that renderer can hide dirt pile
                    dirt_at_position = env.state[nms.DIRT_PILES].by_pos(pos)
                    dirt_at_position[0].set_new_amount(0)

            if self.cfg[nms.ALGORITHM][nms.PILE_ALL_DONE] in [nms.ALL, nms.DISTRIBUTED]:
                if all([all(cleaned_dirt_piles[i].values()) for i in range(self.n_agents)]):
                    done = True
            elif self.cfg[nms.ALGORITHM][nms.PILE_ALL_DONE] == nms.SHARED:
                # End episode if both agents together have cleaned all dirt piles
                if all(get_all_cleaned_dirt_piles(dirt_piles_positions, cleaned_dirt_piles, self.n_agents).values()):
                    done = True

        return reward, done
