import copy
from typing import List

import numpy as np
import torch

from marl_factory_grid.algorithms.rl.base_a2c import cumulate_discount
from marl_factory_grid.algorithms.rl.constants import Names

nms = Names

def _as_torch(x):
    """ Helper function to convert different list types to a torch tensor """
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x)
    elif isinstance(x, List):
        return torch.tensor(x)
    elif isinstance(x, (int, float)):
        return torch.tensor([x])
    return x


def transform_observations(env, ordered_dirt_piles, target_pile, cfg, n_agents):
    """ Function that extracts local observations from global state
    Requires that agents have observations -DirtPiles and -Self (cf. environment configs) """
    agents_positions = get_agents_positions(env, n_agents)
    pile_observability_is_all = cfg[nms.ALGORITHM][nms.PILE_OBSERVABILITY] == nms.ALL
    if pile_observability_is_all:
        trans_obs = [torch.zeros(2+2*len(ordered_dirt_piles[0])) for _ in range(len(agents_positions))]
    else:
        # Only show current target pile
        trans_obs = [torch.zeros(4) for _ in range(len(agents_positions))]
    for i, pos in enumerate(agents_positions):
        agent_x, agent_y = pos[0], pos[1]
        trans_obs[i][0] = agent_x
        trans_obs[i][1] = agent_y
        idx = 2
        if pile_observability_is_all:
            for pile_pos in ordered_dirt_piles[i]:
                trans_obs[i][idx] = pile_pos[0]
                trans_obs[i][idx + 1] = pile_pos[1]
                idx += 2
        else:
            trans_obs[i][2] = ordered_dirt_piles[i][target_pile[i]][0]
            trans_obs[i][3] = ordered_dirt_piles[i][target_pile[i]][1]
    return trans_obs


def get_all_observations(env, cfg, n_agents):
    """ Helper function that returns all possible agent observations """
    dirt_piles_positions = [env.state.entities[nms.DIRT_PILES][pile_idx].pos for pile_idx in
                            range(len(env.state.entities[nms.DIRT_PILES]))]
    if cfg[nms.ALGORITHM][nms.PILE_OBSERVABILITY] == nms.ALL:
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
        obs = [torch.zeros(4) for _ in range(n_agents) for _ in dirt_piles_positions]
        observations = [[] for _ in dirt_piles_positions]
        for idx, pile_pos in enumerate(dirt_piles_positions):
            obs[idx][2] = pile_pos[0]
            obs[idx][3] = pile_pos[1]
    valid_agent_positions = env.state.entities.floorlist

    for idx, pos in enumerate(valid_agent_positions):
        for obs_layer in range(len(obs)):
            observation = copy.deepcopy(obs[obs_layer])
            observation[0] = pos[0]
            observation[1] = pos[1]
            observations[obs_layer].append(observation)

    return observations


def get_dirt_piles_positions(env):
    """ Get positions of dirt piles on the map """
    return [env.state.entities[nms.DIRT_PILES][pile_idx].pos for pile_idx in range(len(env.state.entities[nms.DIRT_PILES]))]


def get_agents_positions(env, n_agents):
    """ Get positions of agents on the map """
    return [env.state.moving_entites[agent_idx].pos for agent_idx in range(n_agents)]


def get_ordered_dirt_piles(env, cleaned_dirt_piles, cfg, n_agents):
    """ This function determines in which order the agents should clean the dirt piles
    Each agent can have its individual pile order """
    ordered_dirt_piles = [[] for _ in range(n_agents)]
    dirt_piles_positions = get_dirt_piles_positions(env)
    agents_positions = get_agents_positions(env, n_agents)
    for agent_idx in range(n_agents):
        if cfg[nms.ALGORITHM][nms.PILE_ORDER] in [nms.FIXED, nms.AGENTS]:
            ordered_dirt_piles[agent_idx] = dirt_piles_positions
        elif cfg[nms.ALGORITHM][nms.PILE_ORDER] in [nms.SMART, nms.DYNAMIC]:
            # Calculate distances for remaining unvisited dirt piles
            remaining_target_piles = [pos for pos, value in cleaned_dirt_piles[agent_idx].items() if not value]
            pile_distances = {pos:0 for pos in remaining_target_piles}
            agent_pos = agents_positions[agent_idx]
            for pos in remaining_target_piles:
                pile_distances[pos] = np.abs(agent_pos[0] - pos[0]) + np.abs(agent_pos[1] - pos[1])

            if cfg[nms.ALGORITHM][nms.PILE_ORDER] == nms.SMART:
                # Check if there is an agent on the direct path to any of the remaining dirt piles
                for pile_pos in remaining_target_piles:
                    for other_pos in agents_positions:
                        if other_pos != agent_pos:
                            if agent_pos[0] == other_pos[0] == pile_pos[0] or agent_pos[1] == other_pos[1] == pile_pos[1]:
                                # Get the line between the agent and the target
                                path = bresenham(agent_pos[0], agent_pos[1], pile_pos[0], pile_pos[1])

                                # Check if the entity lies on the path between the agent and the target
                                if other_pos in path:
                                    pile_distances[pile_pos] += np.abs(agent_pos[0] - other_pos[0]) + np.abs(agent_pos[1] - other_pos[1])

            sorted_pile_distances = dict(sorted(pile_distances.items(), key=lambda item: item[1]))
            # Insert already visited dirt piles
            ordered_dirt_piles[agent_idx] = [pos for pos in dirt_piles_positions if pos not in remaining_target_piles]
            # Fill up with sorted positions
            for pos in sorted_pile_distances.keys():
                ordered_dirt_piles[agent_idx].append(pos)

        else:
            print("Not a valid pile order option.")
            exit()

    return ordered_dirt_piles

def bresenham(x0, y0, x1, y1):
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


def update_ordered_dirt_piles(agent_idx, cleaned_dirt_piles, ordered_dirt_piles, env, cfg, n_agents):
    """ Update the order of the remaining dirt piles """
    # Only update ordered_dirt_pile for agent that reached its target pile
    updated_ordered_dirt_piles = get_ordered_dirt_piles(env, cleaned_dirt_piles, cfg, n_agents)
    for i in range(len(ordered_dirt_piles[agent_idx])):
        ordered_dirt_piles[agent_idx][i] = updated_ordered_dirt_piles[agent_idx][i]


def distribute_indices(env, cfg, n_agents):
    """ Distribute dirt piles evenly among the agents """
    indices = []
    n_dirt_piles = len(get_dirt_piles_positions(env))
    agents_positions = get_agents_positions(env, n_agents)
    if n_dirt_piles == 1 or cfg[nms.ALGORITHM][nms.PILE_ORDER] in [nms.FIXED, nms.DYNAMIC, nms.SMART]:
        indices = [[0] for _ in range(n_agents)]
    else:
        base_count = n_dirt_piles // n_agents
        remainder = n_dirt_piles % n_agents

        start_index = 0
        for i in range(n_agents):
            # Add an extra index to the first 'remainder' objects
            end_index = start_index + base_count + (1 if i < remainder else 0)
            indices.append(list(range(start_index, end_index)))
            start_index = end_index

        # Static form: auxiliary pile, primary pile, auxiliary pile, ...
        # -> Starting with index 0 even piles are auxiliary piles, odd piles are primary piles
        if cfg[nms.ALGORITHM][nms.AUXILIARY_PILES] and nms.DOORS in env.state.entities.keys():
            door_positions = [door.pos for door in env.state.entities[nms.DOORS]]
            distances = {door_pos:[] for door_pos in door_positions}

            # Calculate distance of every agent to every door
            for door_pos in door_positions:
                for agent_pos in agents_positions:
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


def update_target_pile(env, agent_idx, target_pile, indices, cfg):
    """ Get the next target pile for a given agent """
    if cfg[nms.ALGORITHM][nms.PILE_ORDER] in [nms.FIXED, nms.DYNAMIC, nms.SMART]:
        if target_pile[agent_idx] + 1 < len(get_dirt_piles_positions(env)):
            target_pile[agent_idx] += 1
        else:
            target_pile[agent_idx] = 0
    else:
        if target_pile[agent_idx] + 1 in indices[agent_idx]:
            target_pile[agent_idx] += 1


def is_door_close(env, agent_idx):
    """ Checks whether the agent is close to a door """
    neighbourhood = [y for x in env.state.entities.neighboring_positions(env.state[nms.AGENT][agent_idx].pos)
                    for y in env.state.entities.pos_dict[x] if nms.DOOR in y.name]
    if neighbourhood:
        return neighbourhood[0]


def get_all_cleaned_dirt_piles(dirt_piles_positions, cleaned_dirt_piles, n_agents):
    """ Returns all dirt piles cleaned by any agent """
    meta_cleaned_dirt_piles = {pos: False for pos in dirt_piles_positions}
    for agent_idx in range(n_agents):
        for (pos, cleaned) in cleaned_dirt_piles[agent_idx].items():
            if cleaned:
                meta_cleaned_dirt_piles[pos] = True
    return meta_cleaned_dirt_piles


def handle_finished_episode(obs, agents, cfg):
    """ Finish up episode, calculate advantages and perform policy net and value net updates"""
    with torch.inference_mode(False):
        for ag_i, agent in enumerate(agents):
            # Get states, actions, rewards and values from rollout buffer
            data = agent.finish_episode()
            # Chunk episode data, such that there will be no memory failure for very long episodes
            chunks = split_into_chunks(data, cfg)
            for (s, a, R, V) in chunks:
                # Calculate discounted return and advantage
                G = cumulate_discount(R, cfg[nms.ALGORITHM][nms.GAMMA])
                if cfg[nms.ALGORITHM][nms.ADVANTAGE] == nms.REINFORCE:
                    A = G
                elif cfg[nms.ALGORITHM][nms.ADVANTAGE] == nms.ADVANTAGE_AC:
                    A = G - V  # Actor-Critic Advantages
                elif cfg[nms.ALGORITHM][nms.ADVANTAGE] == nms.TD_ADVANTAGE_AC:
                    with torch.no_grad():
                        A = R + cfg[nms.ALGORITHM][nms.GAMMA] * np.append(V[1:], agent.vf(
                            _as_torch(obs[ag_i]).view(-1).to(
                                torch.float32)).numpy()) - V  # TD Actor-Critic Advantages
                else:
                    print("Not a valid advantage option.")
                    exit()

                rollout = (torch.tensor(x.copy()).to(torch.float32) for x in (s, a, G, A))
                # Update policy and value net of agent with experience from rollout buffer
                agent.train(*rollout)


def split_into_chunks(data_tuple, cfg):
    """ Chunks episode data into approximately equal sized chunks to prevent system memory failure from overload """
    result = [data_tuple]
    chunk_size = cfg[nms.ALGORITHM][nms.CHUNK_EPISODE]
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


def set_agents_spawnpoints(env, n_agents):
    """ Tell environment where the agents should spawn in the next episode """
    for agent_idx in range(n_agents):
        agent_name = list(env.state.agents_conf.keys())[agent_idx]
        current_pos_pointer = env.state.agents_conf[agent_name][nms.POS_POINTER]
        # Making the reset dependent on the number of spawnpoints and not the number of dirtpiles allows
        # for having multiple subsequent spawnpoints with the same target pile
        if current_pos_pointer == len(env.state.agents_conf[agent_name][nms.POSITIONS]) - 1:
            env.state.agents_conf[agent_name][nms.POS_POINTER] = 0
        else:
            env.state.agents_conf[agent_name][nms.POS_POINTER] += 1


def save_configs(results_path, cfg, factory_conf, eval_factory_conf):
    """ Save configurations for logging purposes """
    with open(f"{results_path}/MARL_config.txt", "w") as txt_file:
        txt_file.write(str(cfg))
    with open(f"{results_path}/train_env_config.txt", "w") as txt_file:
        txt_file.write(str(factory_conf))
    with open(f"{results_path}/eval_env_config.txt", "w") as txt_file:
        txt_file.write(str(eval_factory_conf))


def save_agent_models(results_path, agents):
    """ Save model parameters after training """
    for idx, agent in enumerate(agents):
        agent.pi.save_model_parameters(results_path)
        agent.vf.save_model_parameters(results_path)
