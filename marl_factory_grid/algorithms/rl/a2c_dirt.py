import os
import pickle
import torch
from typing import Union, List
import numpy as np
import wandb
from tqdm import tqdm

from marl_factory_grid.algorithms.rl.base_a2c import PolicyGradient, cumulate_discount
from marl_factory_grid.algorithms.rl.constants import Names
from marl_factory_grid.algorithms.rl.utils import transform_observations, _as_torch, is_door_close, \
    get_dirt_piles_positions, update_target_pile, update_ordered_dirt_piles, get_all_cleaned_dirt_piles, \
    distribute_indices, set_agents_spawnpoints, get_ordered_dirt_piles, handle_finished_episode, save_configs, \
    save_agent_models, get_all_observations, get_agents_positions, has_low_change_phase_started, significant_deviation, \
    log_wandb_training

from marl_factory_grid.algorithms.utils import add_env_props
from marl_factory_grid.utils.plotting.plot_single_runs import plot_action_maps, plot_return_development, \
    create_info_maps, plot_return_development_change

nms = Names
ListOrTensor = Union[List, torch.Tensor]


class A2C:
    def __init__(self, train_cfg=None, eval_cfg=None, mode="train"):
        self.mode = mode
        if mode == nms.TRAIN:
            self.train_factory = add_env_props(train_cfg)
            self.train_cfg = train_cfg
            self.n_agents = train_cfg[nms.ENV][nms.N_AGENTS]
        else:
            self.n_agents = eval_cfg[nms.ENV][nms.N_AGENTS]
        self.eval_factory = add_env_props(eval_cfg)
        self.eval_cfg = eval_cfg
        self.setup()
        self.action_probabilities = {agent_idx: [] for agent_idx in range(self.n_agents)}

    def setup(self):
        """ Initialize agents and create entry for run results according to configuration """
        if self.mode == "train":
            self.cfg = self.train_cfg
            self.factory = self.train_factory
            self.gamma = self.cfg[nms.ALGORITHM][nms.GAMMA]
        else:
            self.cfg = self.eval_cfg
            self.factory = self.eval_factory
            self.gamma = 0.99

        seed = self.cfg[nms.ALGORITHM][nms.SEED]
        print("Algorithm Seed: ", seed)
        if seed == -1:
            seed = np.random.choice(range(1000))
            print("Algorithm seed is -1. Pick random seed: ", seed)

        self.obs_dim = 2 + 2 * len(get_dirt_piles_positions(self.factory)) if self.cfg[nms.ALGORITHM][
                                                                                  nms.PILE_OBSERVABILITY] == nms.ALL else 4
        self.act_dim = 4  # The 4 movement directions
        self.agents = [PolicyGradient(self.factory, seed=seed, gamma=self.gamma, agent_id=i, obs_dim=self.obs_dim, act_dim=self.act_dim) for i in
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
            if self.cfg[nms.ENV][nms.WANDB_LOG]:
                wandb.login()
                wandb.init(project="EDYS", name=str(next_run_number))

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
        n_steps, max_steps = [self.train_cfg[nms.ALGORITHM][k] for k in [nms.N_STEPS, nms.MAX_STEPS]]
        global_steps, episode = 0, 0
        indices = distribute_indices(env, self.train_cfg, self.n_agents)
        dirt_piles_positions = get_dirt_piles_positions(env)
        target_pile = [partition[0] for partition in
                       indices]  # list of pointers that point to the current target pile for each agent
        cleaned_dirt_piles = [{pos: False for pos in dirt_piles_positions} for _ in range(self.n_agents)]
        low_change_phase_start_episode = -1
        episode_rewards_development = []
        return_change_development = []

        pbar = tqdm(total=max_steps)
        loop_condition = True if self.train_cfg[nms.ALGORITHM][nms.EARLY_STOPPING] else global_steps < max_steps
        while loop_condition:
            _ = env.reset()
            if self.train_cfg[nms.ENV][nms.TRAIN_RENDER]:
                env.render()
            set_agents_spawnpoints(env, self.n_agents)
            ordered_dirt_piles = get_ordered_dirt_piles(env, cleaned_dirt_piles, self.train_cfg, self.n_agents)
            # Reset current target pile at episode begin if all piles have to be cleaned in one episode
            if self.train_cfg[nms.ALGORITHM][nms.PILE_ALL_DONE] == nms.ALL:
                target_pile = [partition[0] for partition in indices]
                cleaned_dirt_piles = [{pos: False for pos in dirt_piles_positions} for _ in range(self.n_agents)]
            episode_rewards_development.append([])

            # Supply each agent with its local observation
            obs = transform_observations(env, ordered_dirt_piles, target_pile, self.train_cfg, self.n_agents)
            done, ep_return = [False] * self.n_agents, 0

            if self.train_cfg[nms.ALGORITHM][nms.EARLY_STOPPING]:
                if len(return_change_development) > self.train_cfg[nms.ALGORITHM][
                    nms.LAST_N_EPISODES] and low_change_phase_start_episode == -1 and has_low_change_phase_started(
                        return_change_development, self.train_cfg[nms.ALGORITHM][nms.LAST_N_EPISODES],
                        self.train_cfg[nms.ALGORITHM][nms.MEAN_TARGET_CHANGE]):
                    low_change_phase_start_episode = len(return_change_development)
                    print(low_change_phase_start_episode)

                # Check if requirements for early stopping are met
                if len(return_change_development) % 50 == 0:
                    print(len(return_change_development))
                if low_change_phase_start_episode != -1 and significant_deviation(return_change_development, low_change_phase_start_episode):
                    print(f"Early Stopping in Episode: {global_steps} because of significant deviation.")
                    break
                if low_change_phase_start_episode != -1 and (len(return_change_development) - low_change_phase_start_episode) >= 1000:
                    print(f"Early Stopping in Episode: {global_steps} because of episode time limit")
                    break
                if low_change_phase_start_episode != -1 and global_steps >= max_steps:
                    print(f"Early Stopping in Episode: {global_steps} because of global steps time limit")
                    break

            while not all(done):
                action = self.use_door_or_move(env, obs, cleaned_dirt_piles) \
                    if nms.DOORS in env.state.entities.keys() else self.get_actions(obs)
                _, next_obs, reward, done, info = env.step(action)
                next_obs = transform_observations(env, ordered_dirt_piles, target_pile, self.train_cfg, self.n_agents)

                # Handle case where agent is on field with dirt
                reward, done = self.handle_dirt(env, cleaned_dirt_piles, ordered_dirt_piles, target_pile, indices,
                                                reward, done, self.train_cfg)

                if n_steps != 0 and (global_steps + 1) % n_steps == 0: done = True

                done = [done] * self.n_agents if isinstance(done, bool) else done
                for ag_i, agent in enumerate(self.agents):
                    if action[ag_i] in range(self.act_dim):
                        # Add agent results into respective rollout buffers
                        agent._episode[-1] = (next_obs[ag_i], action[ag_i], reward[ag_i], agent._episode[-1][-1])

                # Visualize state update
                if self.train_cfg[nms.ENV][nms.TRAIN_RENDER]: env.render()

                obs = next_obs

                if all(done): handle_finished_episode(obs, self.agents, self.train_cfg)

                global_steps += 1
                episode_rewards_development[-1].extend(reward)

                if global_steps >= max_steps: break

            return_change_development.append(
                sum(episode_rewards_development[-1]) - sum(episode_rewards_development[-2])
                if len(episode_rewards_development) > 1 else 0.0)
            episode += 1
            if self.cfg[nms.ENV][nms.SAVE_AND_LOG] and self.train_cfg[nms.ENV][nms.WANDB_LOG]:
                log_wandb_training(episode, sum(episode_rewards_development[-1]),
                                   np.sum([reward * pow(self.gamma, i) for i, reward in enumerate(episode_rewards_development[-1])]),
                                   return_change_development[-1])
            pbar.update(global_steps - pbar.n)

        pbar.close()
        if self.train_cfg[nms.ENV][nms.SAVE_AND_LOG]:
            return_development = [np.sum(rewards) for rewards in episode_rewards_development]
            discounted_return_development = [np.sum([reward * pow(self.gamma, i) for i, reward in enumerate(ep_rewards)]) for ep_rewards in episode_rewards_development]
            plot_return_development(return_development, self.results_path)
            plot_return_development(discounted_return_development, self.results_path, discounted=True)
            plot_return_development_change(return_change_development, self.results_path)
            create_info_maps(env, get_all_observations(env, self.train_cfg, self.n_agents),
                             get_dirt_piles_positions(env), self.results_path, self.agents, self.act_dim, self)
            metrics_data = {"episode_rewards_development": episode_rewards_development,
                            "return_development": return_development,
                            "discounted_return_development": discounted_return_development,
                            "return_change_development": return_change_development}
            with open(f"{self.results_path}/metrics", "wb") as pickle_file:
                pickle.dump(metrics_data, pickle_file)
            save_agent_models(self.results_path, self.agents)
            plot_action_maps(env, [self], self.results_path)

    @torch.inference_mode(True)
    def eval_loop(self, n_episodes):
        """ Function for performing inference """
        env = self.eval_factory
        episode, results = 0, []
        dirt_piles_positions = get_dirt_piles_positions(env)
        print("Dirt Piles positions", dirt_piles_positions)
        indices = distribute_indices(env, self.eval_cfg, self.n_agents)
        target_pile = [partition[0] for partition in
                       indices]  # list of pointers that point to the current target pile for each agent
        if self.eval_cfg[nms.ALGORITHM][nms.PILE_ALL_DONE] == nms.DISTRIBUTED:
            cleaned_dirt_piles = [{dirt_piles_positions[idx]: False for idx in indices[i]} for i in
                                  range(self.n_agents)]
        else:
            cleaned_dirt_piles = [{pos: False for pos in dirt_piles_positions} for _ in range(self.n_agents)]

        cleaned_dirt_piles_per_step = []

        while episode < n_episodes:
            _ = env.reset()
            set_agents_spawnpoints(env, self.n_agents)
            if self.eval_cfg[nms.ENV][nms.EVAL_RENDER]:
                # Don't render auxiliary piles
                if self.eval_cfg[nms.ALGORITHM][nms.AUXILIARY_PILES]:
                    auxiliary_piles = [pile for idx, pile in enumerate(env.state.entities[nms.DIRT_PILES]) if
                                       idx % 2 == 0]
                    for pile in auxiliary_piles:
                        pile.set_new_amount(0)
                env.render()
                env._renderer.fps = 5  # Slow down agent movement

            # Reset current target pile at episode begin if all piles have to be cleaned in one episode
            if self.eval_cfg[nms.ALGORITHM][nms.PILE_ALL_DONE] in [nms.ALL, nms.DISTRIBUTED, nms.SHARED]:
                target_pile = [partition[0] for partition in indices]
                if self.eval_cfg[nms.ALGORITHM][nms.PILE_ALL_DONE] == nms.DISTRIBUTED:
                    cleaned_dirt_piles = [{dirt_piles_positions[idx]: False for idx in indices[i]} for i in
                                          range(self.n_agents)]
                else:
                    cleaned_dirt_piles = [{pos: False for pos in dirt_piles_positions} for _ in range(self.n_agents)]

            ordered_dirt_piles = get_ordered_dirt_piles(env, cleaned_dirt_piles, self.eval_cfg, self.n_agents)

            # Supply each agent with its local observation
            obs = transform_observations(env, ordered_dirt_piles, target_pile, self.eval_cfg, self.n_agents)
            done, rew_log, eps_rew = [False] * self.n_agents, 0, torch.zeros(self.n_agents)

            cleaned_dirt_piles_per_step.append([])

            ep_steps = 0
            while not all(done):
                action = self.use_door_or_move(env, obs, cleaned_dirt_piles, det=True) \
                    if nms.DOORS in env.state.entities.keys() else self.execute_policy(obs, env,
                                                                                       cleaned_dirt_piles)  # zero exploration
                _, next_obs, reward, done, info = env.step(action)

                # Handle case where agent is on field with dirt
                reward, done = self.handle_dirt(env, cleaned_dirt_piles, ordered_dirt_piles, target_pile, indices,
                                                reward, done, self.eval_cfg)

                ordered_dirt_piles = get_ordered_dirt_piles(env, cleaned_dirt_piles, self.eval_cfg, self.n_agents)

                # Get transformed next_obs that might have been updated because of handle_dirt
                next_obs = transform_observations(env, ordered_dirt_piles, target_pile, self.eval_cfg, self.n_agents)

                done = [done] * self.n_agents if isinstance(done, bool) else done

                if self.eval_cfg[nms.ENV][nms.EVAL_RENDER]: env.render()

                obs = next_obs

                # Count the overall number of cleaned dirt piles in each step
                cleaned_piles = 0
                for dict in cleaned_dirt_piles:
                    for value in dict.values():
                        if value:
                            cleaned_piles += 1
                cleaned_dirt_piles_per_step[-1].append(cleaned_piles)

                ep_steps += 1

            episode += 1
            print(ep_steps)
            print(cleaned_dirt_piles_per_step)

        if self.eval_cfg[nms.ENV][nms.SAVE_AND_LOG]:
            metrics_data = {"cleaned_dirt_piles_per_step": cleaned_dirt_piles_per_step}
            with open(f"{self.results_path}/metrics", "wb") as pickle_file:
                pickle.dump(metrics_data, pickle_file)

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

    def handle_dirt(self, env, cleaned_dirt_piles, ordered_dirt_piles, target_pile, indices, reward, done, cfg):
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
                            update_target_pile(env, idx, target_pile, indices, cfg)
                            update_ordered_dirt_piles(idx, cleaned_dirt_piles, ordered_dirt_piles, env,
                                                      cfg, self.n_agents)
                            if cfg[nms.ALGORITHM][nms.PILE_ALL_DONE] == nms.SINGLE:
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

            if cfg[nms.ALGORITHM][nms.PILE_ALL_DONE] in [nms.ALL, nms.DISTRIBUTED]:
                if all([all(cleaned_dirt_piles[i].values()) for i in range(self.n_agents)]):
                    done = True
            elif cfg[nms.ALGORITHM][nms.PILE_ALL_DONE] == nms.SHARED:
                # End episode if both agents together have cleaned all dirt piles
                if all(get_all_cleaned_dirt_piles(dirt_piles_positions, cleaned_dirt_piles, self.n_agents).values()):
                    done = True

        return reward, done
