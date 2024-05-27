import ast
import os
import pickle
from os import PathLike
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt

from marl_factory_grid.algorithms.rl.utils import _as_torch
from marl_factory_grid.utils.helpers import IGNORED_DF_COLUMNS

from marl_factory_grid.utils.renderer import Renderer
from marl_factory_grid.utils.utility_classes import RenderEntity

from marl_factory_grid.modules.clean_up import constants as d


def plot_routes(factory, agents):
    """
    Creates a plot of the agents' actions on the level map by creating a Renderer and Render Entities that hold the
    icon that corresponds to the action. For deterministic agents, simply displays the agents path of actions while for
    RL agents that can supply an action map or action probabilities from their policy net.
    """
    renderer = Renderer(factory.map.level_shape, custom_assets_path={
        'cardinal': 'marl_factory_grid/utils/plotting/action_assets/cardinal.png',
        'diagonal': 'marl_factory_grid/utils/plotting/action_assets/diagonal.png',
        'use_door': 'marl_factory_grid/utils/plotting/action_assets/door_action.png',
        'wall': 'marl_factory_grid/environment/assets/wall.png',
        'machine_action': 'marl_factory_grid/utils/plotting/action_assets/machine_action.png',
        'clean_action': 'marl_factory_grid/utils/plotting/action_assets/clean_action.png',
        'destination_action': 'marl_factory_grid/utils/plotting/action_assets/destination_action.png',
        'noop': 'marl_factory_grid/utils/plotting/action_assets/noop.png',
        'charge_action': 'marl_factory_grid/utils/plotting/action_assets/charge_action.png'})

    wall_positions = swap_coordinates(factory.map.walls)
    wall_entities = [RenderEntity(name='wall', probability=0, pos=np.array(pos)) for pos in wall_positions]
    action_entities = list(wall_entities)

    for index, agent in enumerate(agents):
        current_position = swap_coordinates(agent.spawn_position)

        if hasattr(agent, 'action_probabilities'):
            # Handle RL agents with action probabilities
            top_actions = sorted(agent.action_probabilities.items(), key=lambda x: -x[1])[:4]
        else:
            # Handle deterministic agents by iterating through all actions in the list
            top_actions = [(action, 0) for action in agent.action_list]

        for action, probability in top_actions:
            if action.lower() in rotation_mapping:
                base_icon, rotation = rotation_mapping[action.lower()]
                icon_name = 'cardinal' if 'diagonal' not in base_icon else 'diagonal'
                new_position = action_to_coords(current_position, action.lower())
            else:
                icon_name = action.lower()
                rotation = 0
                new_position = current_position

            action_entity = RenderEntity(
                name=icon_name,
                pos=np.array(current_position),
                probability=probability,
                rotation=rotation
            )
            action_entities.append(action_entity)
            current_position = new_position

    renderer.render_single_action_icons(action_entities)  # move in/out loop for graph per agent or not


def plot_action_maps(factory, agents, result_path):
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    assets_path = {
        'green_arrow': os.path.join(base_dir, 'utils', 'plotting', 'action_assets', 'green_arrow.png'),
        'yellow_arrow': os.path.join(base_dir, 'utils', 'plotting', 'action_assets', 'yellow_arrow.png'),
        'red_arrow': os.path.join(base_dir, 'utils', 'plotting', 'action_assets', 'red_arrow.png'),
        'grey_arrow': os.path.join(base_dir, 'utils', 'plotting', 'action_assets', 'grey_arrow.png'),
        'wall': os.path.join(base_dir, 'environment', 'assets', 'wall.png'),
        'target_dirt': os.path.join(base_dir, 'utils', 'plotting', 'action_assets', 'target_dirt.png'),
        'spawn_pos': os.path.join(base_dir, 'utils', 'plotting', 'action_assets', 'spawn_pos.png')
    }
    renderer = Renderer(factory.map.level_shape, cell_size=80, custom_assets_path=assets_path)

    directions = ['north', 'east', 'south', 'west']
    wall_positions = swap_coordinates(factory.map.walls)

    for agent_index, agent in enumerate(agents):
        if hasattr(agent, 'action_probabilities'):
            action_probabilities = unpack_action_probabilities(agent.action_probabilities)
            for action_map_index, probabilities_map in enumerate(action_probabilities[agent_index]):

                wall_entities = [RenderEntity(name='wall', probability=0, pos=np.array(pos)) for pos in wall_positions]
                action_entities = list(wall_entities)
                target_dirt_pos = factory.state.entities[d.DIRT][action_map_index].pos
                action_entities.append(
                    RenderEntity(name='target_dirt', probability=0, pos=swap_coordinates(target_dirt_pos)))

                # Render all spawnpoints assigned to current target dirt pile
                spawnpoints = list(factory.state.agents_conf.values())[agent_index]['positions']
                all_target_dirts = []
                if 'DirtPiles' in factory.conf['Entities']:
                    tuples = ast.literal_eval(factory.conf['Entities']['DirtPiles']['coords_or_quantity'])
                    for t in tuples:
                        all_target_dirts.append(t)
                assigned_spawn_positions = []
                for j in range(len(spawnpoints) // len(all_target_dirts)):
                    assigned_spawn_positions.append(spawnpoints[j * len(all_target_dirts) + all_target_dirts.index(target_dirt_pos)])
                for spawn_pos in assigned_spawn_positions:
                    action_entities.append(RenderEntity(name='spawn_pos', probability=0, pos=swap_coordinates(spawn_pos)))

                render_arrows = []
                for position, probabilities in probabilities_map.items():
                    if position not in wall_positions:
                        if np.any(probabilities) > 0:  # Ensure it's not all zeros which would indicate a wall
                            sorted_indices = np.argsort(np.argsort(-probabilities))
                            colors = ['green_arrow', 'yellow_arrow', 'red_arrow', 'grey_arrow']
                            render_arrows.append([])
                            for rank, direction_index in enumerate(sorted_indices):
                                action = directions[direction_index]
                                probability = probabilities[rank]
                                arrow_color = colors[direction_index]
                                render_arrows[-1].append((probability, arrow_color, position))

                # Swap west and east
                for l in render_arrows:
                    l[1], l[3] = l[3], l[1]
                for l in render_arrows:
                    for rank, (probability, arrow_color, position) in enumerate(l):
                        if probability > 0:
                            action_entity = RenderEntity(
                                name=arrow_color,
                                pos=position,
                                probability=probability,
                                rotation=rank * 90
                            )
                            action_entities.append(action_entity)

                renderer.render_multi_action_icons(action_entities, result_path)


def unpack_action_probabilities(action_probabilities):
    unpacked = {}
    for agent_index, maps in action_probabilities.items():
        unpacked[agent_index] = []
        for map_index, probability_map in enumerate(maps):
            single_map = {}
            for y in range(len(probability_map)):
                for x in range(len(probability_map[y])):
                    position = (x, y)
                    probabilities = probability_map[y][x]
                    single_map[position] = probabilities
            unpacked[agent_index].append(single_map)
    return unpacked


def swap_coordinates(positions):
    """
    Swaps x and y coordinates of single positions, lists or arrays
    """
    if isinstance(positions, tuple) or (isinstance(positions, list) and len(positions) == 2):
        return positions[1], positions[0]
    elif isinstance(positions, np.ndarray) and positions.ndim == 1 and positions.shape[0] == 2:
        return positions[1], positions[0]
    else:
        return [(y, x) for x, y in positions]


def action_to_coords(current_position, action):
    """
    Calculates new coordinates based on the current position and a movement action.
    """
    delta = direction_mapping.get(action)
    if delta is not None:
        new_position = [current_position[0] + delta[0], current_position[1] + delta[1]]
        return new_position
    print(f"No valid movement action found for {action}.")
    return current_position


rotation_mapping = {
    'north': ('cardinal', 0),
    'east': ('cardinal', 270),
    'south': ('cardinal', 180),
    'west': ('cardinal', 90),
    'north_east': ('diagonal', 0),
    'south_east': ('diagonal', 270),
    'south_west': ('diagonal', 180),
    'north_west': ('diagonal', 90)
}

direction_mapping = {
    'north': (0, -1),
    'south': (0, 1),
    'east': (1, 0),
    'west': (-1, 0),
    'north_east': (1, -1),
    'north_west': (-1, -1),
    'south_east': (1, 1),
    'south_west': (-1, 1)
}


def plot_reward_development(reward_development, results_path):
    smoothed_data = np.convolve(reward_development, np.ones(10) / 10, mode='valid')
    plt.plot(smoothed_data)
    plt.ylim([-10, max(smoothed_data) + 20])
    plt.title('Smoothed Reward Development')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.savefig(f"{results_path}/smoothed_reward_development.png")
    plt.show()


def plot_collected_coins_per_step():
    # Observed behaviour for multi-agent setting consisting of run0 and run0
    cleaned_dirt_per_step_emergent = [0, 0, 0, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 5]
    cleaned_dirt_per_step = [0, 0, 0, 1, 1, 2, 2, 3, 3, 3, 4, 5] # RL and TSP

    plt.step(range(1, len(cleaned_dirt_per_step) + 1), cleaned_dirt_per_step, color='green', linewidth=3, label='Prevented (RL)')
    plt.step(range(1, len(cleaned_dirt_per_step_emergent) + 1), cleaned_dirt_per_step_emergent, linestyle='--', color='darkred', linewidth=3, label='Emergent')
    plt.step(range(1, len(cleaned_dirt_per_step) + 1), cleaned_dirt_per_step, linestyle='dotted', color='darkorange', linewidth=3, label='Prevented (TSP)')
    plt.xlabel("Environment step", fontsize=20)
    plt.ylabel("Collected Coins", fontsize=20)
    yint = range(min(cleaned_dirt_per_step), max(cleaned_dirt_per_step) + 1)
    plt.yticks(yint, fontsize=17)
    plt.xticks(range(1, len(cleaned_dirt_per_step_emergent) + 1), fontsize=17)
    frame1 = plt.gca()
    # Only display every 5th tick label
    for idx, xlabel_i in enumerate(frame1.axes.get_xticklabels()):
        if (idx + 1) % 5 != 0:
            xlabel_i.set_visible(False)
            xlabel_i.set_fontsize(0.0)
    # Change order of labels in legend
    handles, labels = frame1.get_legend_handles_labels()
    order = [0, 2, 1]
    plt.legend([handles[idx] for idx in order], [labels[idx] for idx in order], prop={'size': 20})
    fig = plt.gcf()
    fig.set_size_inches(8, 7)
    plt.savefig("../study_out/number_of_collected_coins.pdf")
    plt.show()


def plot_reached_flags_per_step():
    # Observed behaviour for multi-agent setting consisting of runs 1 + 2
    reached_flags_per_step_emergent = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    reached_flags_per_step_RL = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2]
    reached_flags_per_step_TSP = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2]

    plt.step(range(1, len(reached_flags_per_step_RL) + 1), reached_flags_per_step_RL, color='green', linewidth=3, label='Prevented (RL)')
    plt.step(range(1, len(reached_flags_per_step_emergent) + 1), reached_flags_per_step_emergent,  linestyle='--', color='darkred', linewidth=3, label='Emergent')
    plt.step(range(1, len(reached_flags_per_step_TSP) + 1), reached_flags_per_step_TSP, linestyle='dotted', color='darkorange', linewidth=3, label='Prevented (TSP)')
    plt.xlabel("Environment step", fontsize=20)
    plt.ylabel("Reached Flags", fontsize=20)
    yint = range(min(reached_flags_per_step_RL), max(reached_flags_per_step_RL) + 1)
    plt.yticks(yint, fontsize=17)
    plt.xticks(range(1, len(reached_flags_per_step_emergent) + 1), fontsize=17)
    frame1 = plt.gca()
    # Only display every 5th tick label
    for idx, xlabel_i in enumerate(frame1.axes.get_xticklabels()):
        if (idx + 1) % 5 != 0:
            xlabel_i.set_visible(False)
            xlabel_i.set_fontsize(0.0)
    # Change order of labels in legend
    handles, labels = frame1.get_legend_handles_labels()
    order = [0, 2, 1]
    plt.legend([handles[idx] for idx in order], [labels[idx] for idx in order], prop={'size': 20})
    fig = plt.gcf()
    fig.set_size_inches(8, 7)
    plt.savefig("../study_out/number_of_reached_flags.pdf")
    plt.show()


def create_info_maps(env, all_valid_observations, dirt_piles_positions, results_path, agents, act_dim,
                     a2c_instance):
    # Create value map
    with open(f"{results_path}/info_maps.txt", "w") as txt_file:
        for obs_layer, pos in enumerate(dirt_piles_positions):
            observations_shape = (
                max(t[0] for t in env.state.entities.floorlist) + 2,
                max(t[1] for t in env.state.entities.floorlist) + 2)
            value_maps = [np.zeros(observations_shape) for _ in agents]
            likeliest_action = [np.full(observations_shape, np.NaN) for _ in agents]
            action_probabilities = [np.zeros((observations_shape[0], observations_shape[1], act_dim)) for
                                    _ in agents]
            for obs in all_valid_observations[obs_layer]:
                for idx, agent in enumerate(agents):
                    x, y = int(obs[0]), int(obs[1])
                    try:
                        value_maps[idx][x][y] = agent.vf(obs)
                        probs = agent.pi.distribution(obs).probs
                        likeliest_action[idx][x][y] = torch.argmax(
                            probs)  # get the likeliest action at the current agent position
                        action_probabilities[idx][x][y] = probs
                    except:
                        pass

            txt_file.write("=======Value Maps=======\n")
            for agent_idx, vmap in enumerate(value_maps):
                txt_file.write(f"Value map of agent {agent_idx} for target pile {pos}:\n")
                vmap = _as_torch(vmap).round(decimals=4)
                max_digits = max(len(str(vmap.max().item())), len(str(vmap.min().item())))
                for idx, row in enumerate(vmap):
                    txt_file.write(' '.join(f" {elem:>{max_digits + 1}}" for elem in row.tolist()))
                    txt_file.write("\n")
            txt_file.write("\n")
            txt_file.write("=======Likeliest Action=======\n")
            for agent_idx, amap in enumerate(likeliest_action):
                txt_file.write(f"Likeliest action map of agent {agent_idx} for target pile {pos}:\n")
                txt_file.write(np.array2string(amap))
            txt_file.write("\n")
            txt_file.write("=======Action Probabilities=======\n")
            for agent_idx, pmap in enumerate(action_probabilities):
                a2c_instance.action_probabilities[agent_idx].append(pmap)
                txt_file.write(f"Action probability map of agent {agent_idx} for target pile {pos}:\n")
                for d in range(pmap.shape[0]):
                    row = '['
                    for r in range(pmap.shape[1]):
                        row += "[" + ', '.join(f"{x:7.4f}" for x in pmap[d, r]) + "]"
                    txt_file.write(row + "]")
                    txt_file.write("\n")

    return action_probabilities
