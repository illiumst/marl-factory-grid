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
import scipy.stats as stats

from marl_factory_grid.algorithms.marl.utils import _as_torch
from marl_factory_grid.utils.helpers import IGNORED_DF_COLUMNS
from marl_factory_grid.utils.plotting.plotting_utils import prepare_plot

from marl_factory_grid.utils.renderer import Renderer
from marl_factory_grid.utils.utility_classes import RenderEntity

from marl_factory_grid.modules.coins import constants as c

def plot_single_run(run_path: Union[str, PathLike], use_tex: bool = False, column_keys=None,
                    file_key: str = 'monitor', file_ext: str = 'pkl'):
    """
    Plots the Epoch score (step reward)  over a single run based on monitoring data stored in a file.

    :param run_path: The path to the directory containing monitoring data or directly to the monitoring file.
    :type run_path: Union[str, PathLike]
    :param use_tex: Flag indicating whether to use TeX for plotting.
    :type use_tex: bool, optional
    :param column_keys: Specific columns to include in the plot. If None, includes all columns except ignored ones.
    :type column_keys: list or None, optional
    :param file_key: The keyword to identify the monitoring file.
    :type file_key: str, optional
    :param file_ext: The extension of the monitoring file.
    :type file_ext: str, optional
    """
    run_path = Path(run_path)
    df_list = list()
    if run_path.is_dir():
        monitor_file = next(run_path.glob(f'*{file_key}*.{file_ext}'))
    elif run_path.exists() and run_path.is_file():
        monitor_file = run_path
    else:
        raise ValueError

    with monitor_file.open('rb') as f:
        monitor_df = pickle.load(f)

        monitor_df = monitor_df.fillna(0)
        df_list.append(monitor_df)

    df = pd.concat(df_list, ignore_index=True)
    df = df.fillna(0).rename(columns={'episode': 'Episode'}).sort_values(['Episode'])
    if column_keys is not None:
        columns = [col for col in column_keys if col in df.columns]
    else:
        columns = [col for col in df.columns if col not in IGNORED_DF_COLUMNS]

    # roll_n = 50
    # non_overlapp_window = df.groupby(['Episode']).rolling(roll_n, min_periods=1).mean()

    df_melted = df[columns + ['Episode']].reset_index().melt(
        id_vars=['Episode'], value_vars=columns, var_name="Measurement", value_name="Score"
    )

    if df_melted['Episode'].max() > 800:
        skip_n = round(df_melted['Episode'].max() * 0.02)
        df_melted = df_melted[df_melted['Episode'] % skip_n == 0]

    prepare_plot(run_path.parent / f'{run_path.parent.name}_monitor_lineplot.png', df_melted, use_tex=use_tex)
    print('Plotting done.')

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
        'target_coin': os.path.join(base_dir, 'utils', 'plotting', 'action_assets', 'target_coin.png'),
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
                target_coin_pos = factory.state.entities[c.COIN][action_map_index].pos
                action_entities.append(
                    RenderEntity(name='target_coin', probability=0, pos=swap_coordinates(target_coin_pos)))

                # Render all spawnpoints assigned to current target coin pile
                spawnpoints = list(factory.state.agents_conf.values())[agent_index]['positions']
                all_target_coins = []
                if 'CoinPiles' in factory.conf['Entities']:
                    tuples = ast.literal_eval(factory.conf['Entities']['CoinPiles']['coords_or_quantity'])
                    for t in tuples:
                        all_target_coins.append(t)

                if isinstance(all_target_coins[0], int):
                    temp = all_target_coins
                    all_target_coins = [tuple(temp)]

                assigned_spawn_positions = []
                for j in range(len(spawnpoints) // len(all_target_coins)):
                    assigned_spawn_positions.append(spawnpoints[j * len(all_target_coins) + all_target_coins.index(target_coin_pos)])
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


def plot_return_development(return_development, results_path, discounted=False):
    smoothed_data = np.convolve(return_development, np.ones(10) / 10, mode='valid')
    plt.plot(smoothed_data)
    plt.ylim([-10, max(smoothed_data) + 20])
    plt.title('Smoothed Return Development' if not discounted else 'Smoothed Discounted Return Development')
    plt.xlabel('Episode')
    plt.ylabel('Return' if not discounted else "Discounted Return")
    plt.savefig(f"{results_path}/smoothed_return_development.png"
                if not discounted else f"{results_path}/smoothed_discounted_return_development.png")
    plt.show()

def plot_return_development_change(return_change_development, results_path):
    plt.plot(return_change_development)
    plt.title('Return Change Development')
    plt.xlabel('Episode')
    plt.ylabel('Delta Return')
    plt.savefig(f"{results_path}/return_change_development.png")
    plt.show()


def mean_confidence_interval(data, confidence=0.95):
    a = np.array(data)
    n = np.sum(~np.isnan(a), axis=0)
    mean = np.nanmean(a, axis=0)
    se = np.nanstd(a, axis=0) / np.sqrt(n)
    h = se * 1.96  # For 95% confidence interval
    return mean, mean - h, mean + h

def load_metrics(file_path, key):
    with open(file_path, "rb") as pickle_file:
        metrics = pickle.load(pickle_file)
    return metrics[key][0]

def pad_runs(runs):
    max_length = max(len(run) for run in runs)
    padded_runs = [np.pad(np.array(run, dtype=float), (0, max_length - len(run)), constant_values=np.nan) for run in runs]
    return padded_runs

def get_reached_flags_metrics(runs):
    # Find the step where flag 1 and flag 2 are reached
    flag1_steps = []
    flag2_steps = []

    for run in runs:
        if 1 in run:
            flag1_steps.append(run.index(1))
        if 2 in run:
            flag2_steps.append(run.index(2))

    print(flag1_steps)
    print(flag2_steps)

    # Calculate the mean steps and confidence intervals
    mean_flag1_steps = np.mean(flag1_steps)
    mean_flag2_steps = np.mean(flag2_steps)

    std_flag1_steps = np.std(flag1_steps, ddof=1)
    std_flag2_steps = np.std(flag2_steps, ddof=1)

    n_flag1 = len(flag1_steps)
    n_flag2 = len(flag2_steps)

    confidence_level = 0.95
    t_critical_flag1 = stats.t.ppf((1 + confidence_level) / 2, n_flag1 - 1)
    t_critical_flag2 = stats.t.ppf((1 + confidence_level) / 2, n_flag2 - 1)

    margin_of_error_flag1 = t_critical_flag1 * (std_flag1_steps / np.sqrt(n_flag1))
    margin_of_error_flag2 = t_critical_flag2 * (std_flag2_steps / np.sqrt(n_flag2))

    # Mean steps including baseline
    mean_steps = [0, mean_flag1_steps, mean_flag2_steps]
    flags_reached = [0, 1, 2]
    error_bars = [0, margin_of_error_flag1, margin_of_error_flag2]
    return mean_steps, flags_reached, error_bars


def plot_collected_coins_per_step(rl_runs_names, tsp_runs_names, results_path):
    # Observed behaviour for multi-agent setting consisting of run0 and run0
    collected_coins_per_step_emergent = [0, 0, 0, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 5]

    # Load RL and TSP data from multiple runs
    rl_runs = [load_metrics(results_path + f"/{rl_run}/metrics", "cleaned_dirt_piles_per_step") for rl_run in rl_runs_names]

    tsp_runs = [load_metrics(results_path + f"/{tsp_run}/metrics", "cleaned_dirt_piles_per_step") for tsp_run in tsp_runs_names]

    # Pad runs to handle heterogeneous lengths
    rl_runs = pad_runs(rl_runs)
    tsp_runs = pad_runs(tsp_runs)

    # Calculate mean and confidence intervals
    mean_rl, lower_rl, upper_rl = mean_confidence_interval(rl_runs)
    mean_tsp, lower_tsp, upper_tsp = mean_confidence_interval(tsp_runs)

    # Plot the mean and confidence intervals
    plt.fill_between(range(1, len(mean_rl) + 1), lower_rl, upper_rl, color='green', alpha=0.2)
    plt.step(range(1, len(mean_rl) + 1), mean_rl, color='green', linewidth=3, label='Prevented (RL)')

    plt.fill_between(range(1, len(mean_tsp) + 1), lower_tsp, upper_tsp, color='darkorange', alpha=0.2)
    plt.step(range(1, len(mean_tsp) + 1), mean_tsp, linestyle='dotted', color='darkorange', linewidth=3, label='Prevented (TSP)')

    plt.step(range(1, len(collected_coins_per_step_emergent) + 1), collected_coins_per_step_emergent, linestyle='--', color='darkred', linewidth=3, label='Emergent')

    plt.xlabel("Environment step", fontsize=20)
    plt.ylabel("Collected Coins", fontsize=20)
    plt.xticks(range(1, len(collected_coins_per_step_emergent) + 1), fontsize=17)
    plt.yticks(fontsize=17)

    frame1 = plt.gca()
    for idx, xlabel_i in enumerate(frame1.axes.get_xticklabels()):
        if (idx + 1) % 5 != 0:
            xlabel_i.set_visible(False)
            xlabel_i.set_fontsize(0.0)

    handles, labels = frame1.get_legend_handles_labels()
    order = [0, 2, 1]
    plt.legend([handles[idx] for idx in order], [labels[idx] for idx in order], prop={'size': 20})

    fig = plt.gcf()
    fig.set_size_inches(8, 7)
    plt.savefig(f"{results_path}/number_of_collected_coins.pdf")
    plt.show()


def plot_reached_flags_per_step(rl_runs_names, tsp_runs_names, results_path):
    reached_flags_per_step_emergent = [0] * 32  # Adjust based on your data length

    # Load RL and TSP data from multiple runs
    rl_runs = [load_metrics(results_path + f"/{rl_run}/metrics", "cleaned_dirt_piles_per_step") for rl_run in rl_runs_names]
    rl_runs = [[pile - 1 for pile in run] for run in rl_runs]  # Subtract the auxiliary pile

    tsp_runs = [load_metrics(results_path + f"/{tsp_run}/metrics", "reached_flags") for tsp_run in tsp_runs_names]

    # Pad runs to handle heterogeneous lengths
    rl_runs = pad_runs(rl_runs)
    tsp_runs = pad_runs(tsp_runs)

    # Calculate mean and confidence intervals
    mean_rl, lower_rl, upper_rl = mean_confidence_interval(rl_runs)
    mean_tsp, lower_tsp, upper_tsp = mean_confidence_interval(tsp_runs)

    # Plot the mean and confidence intervals
    plt.fill_between(range(1, len(mean_rl) + 1), lower_rl, upper_rl, color='green', alpha=0.2)
    plt.step(range(1, len(mean_rl) + 1), mean_rl, color='green', linewidth=3, label='Prevented (RL)')

    plt.fill_between(range(1, len(mean_tsp) + 1), lower_tsp, upper_tsp, color='darkorange', alpha=0.2)
    plt.step(range(1, len(mean_tsp) + 1), mean_tsp, linestyle='dotted', color='darkorange', linewidth=3, label='Prevented (TSP)')

    plt.step(range(1, len(reached_flags_per_step_emergent) + 1), reached_flags_per_step_emergent, linestyle='--', color='darkred', linewidth=3, label='Emergent')

    plt.xlabel("Environment step", fontsize=20)
    plt.ylabel("Reached Flags", fontsize=20)
    plt.xticks(range(1, len(reached_flags_per_step_emergent) + 1), fontsize=17)
    plt.yticks(fontsize=17)

    frame1 = plt.gca()
    for idx, xlabel_i in enumerate(frame1.axes.get_xticklabels()):
        if (idx + 1) % 5 != 0:
            xlabel_i.set_visible(False)
            xlabel_i.set_fontsize(0.0)

    handles, labels = frame1.get_legend_handles_labels()
    order = [0, 2, 1]
    plt.legend([handles[idx] for idx in order], [labels[idx] for idx in order], prop={'size': 20})

    fig = plt.gcf()
    fig.set_size_inches(8, 7)
    plt.savefig(f"{results_path}/number_of_reached_flags.pdf")
    plt.show()


def plot_performance_distribution_on_coin_quadrant(dirt_quadrant, results_path, grid=False):
    plt.rcParams["figure.autolayout"] = True
    plt.rcParams["axes.edgecolor"] = "black"
    plt.rcParams["axes.linewidth"] = 5.0
    fig = plt.figure(figsize=(18, 13))

    rl_color = '#5D3A9B'
    tsp_color = '#E66100'

    # Boxplot
    boxprops = dict(linestyle='-', linewidth=4)
    whiskerprops = dict(linestyle='-', linewidth=4)
    capprops = dict(linestyle='-', linewidth=4)
    flierprops = dict(marker='o', markersize=14, markeredgewidth=4,
                      linestyle='none')
    medianprops = dict(linestyle='-', linewidth=4, color='#40B0A6')
    meanpointprops = dict(marker='D', markeredgecolor='black',
                          markerfacecolor='firebrick')
    meanlineprops = dict(linestyle='-.', linewidth=4, color='purple')

    bp = plt.boxplot([dirt_quadrant["RL_emergence"], dirt_quadrant["RL_prevented"], dirt_quadrant["TSP_emergence"],
            dirt_quadrant["TSP_prevented"]], patch_artist=True, widths=0.6, flierprops=flierprops,
                    boxprops=boxprops, medianprops=medianprops, meanprops=meanlineprops,
                    whiskerprops=whiskerprops, capprops=capprops,
                    meanline=True, showmeans=False, positions=[1, 2.5, 4, 5.5])

    colors = [rl_color, rl_color, tsp_color, tsp_color]

    for bplot, color in zip([bp], [colors, colors]):
        for patch, color in zip(bplot['boxes'], color):
            patch.set_facecolor(color)

    plt.tick_params(width=5, length=10)
    plt.xticks([1, 2.5, 4, 5.5], labels=['Emergent \n (RL)', 'Prevented \n (RL)', 'Emergent \n (TSP)', 'Prevented \n (TSP)'], fontsize=50)
    plt.yticks(fontsize=50)
    plt.ylabel('No. environment steps', fontsize=50)
    plt.xlabel("Agent Types", fontsize=50)
    plt.grid(grid)
    plt.tight_layout()
    plt.savefig(f"{results_path}/number_of_collected_coins_distribution{'_grid' if grid else ''}.pdf")
    plt.show()

def plot_reached_flags_per_step_with_error(mean_steps_RL_prevented, error_bars_RL_prevented,
                                           mean_steps_TSP_prevented, error_bars_TSP_prevented, flags_reached,
                                           results_path, grid=False):
    plt.rcParams["figure.autolayout"] = True
    plt.rcParams["axes.edgecolor"] = "black"
    plt.rcParams["axes.linewidth"] = 5.0
    fig = plt.figure(figsize=(18, 13))

    # Line plot with error bars
    plt.plot(range(30), [0 for _ in range(30)], color='gray', linestyle='--', linewidth=7,
             label='Emergent')
    plt.errorbar(mean_steps_RL_prevented, flags_reached, xerr=error_bars_RL_prevented, fmt='-o', ecolor='r', capsize=10, capthick=5,
                 markersize=20, label='Prevented (RL) + CI', color='#5D3A9B', linewidth=7)
    plt.errorbar(mean_steps_TSP_prevented, flags_reached, xerr=error_bars_TSP_prevented, fmt='-o', ecolor='r', capsize=10, capthick=5,
                 markersize=20, label='Prevented (TSP) + CI', color='#E66100', linewidth=7)
    plt.tick_params(width=5, length=10)
    plt.xticks(fontsize=50)
    plt.yticks(flags_reached, fontsize=50)
    plt.xlabel("Avg. environment step", fontsize=50)
    plt.ylabel('Reached flags', fontsize=50)
    plt.legend(fontsize=45, loc='best', bbox_to_anchor=(0.38, 0.38))
    plt.grid(grid)
    plt.savefig(f"{results_path}/number_of_reached_flags{'_grid' if grid else ''}.pdf")
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
