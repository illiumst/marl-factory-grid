import json
import pickle
from os import PathLike
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd

from marl_factory_grid.utils.helpers import IGNORED_DF_COLUMNS
from marl_factory_grid.utils.plotting.plotting_utils import prepare_plot

from marl_factory_grid.utils.renderer import Renderer
from marl_factory_grid.utils.utility_classes import RenderEntity


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
            top_actions = [(action, 1.0) for action in agent.action_list]

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


def plot_action_maps(factory, agents):
    renderer = Renderer(factory.map.level_shape, custom_assets_path={
        'green_arrow': 'marl_factory_grid/utils/plotting/action_assets/green_arrow.png',
        'yellow_arrow': 'marl_factory_grid/utils/plotting/action_assets/yellow_arrow.png',
        'red_arrow': 'marl_factory_grid/utils/plotting/action_assets/red_arrow.png',
        'grey_arrow': 'marl_factory_grid/utils/plotting/action_assets/grey_arrow.png',
        'wall': 'marl_factory_grid/environment/assets/wall.png',
    })

    directions = ['north', 'east', 'south', 'west']
    wall_positions = swap_coordinates(factory.map.walls)
    wall_entities = [RenderEntity(name='wall', probability=0, pos=np.array(pos)) for pos in wall_positions]
    action_entities = list(wall_entities)

    dummy_action_map = load_action_map("example_action_map.txt")
    for agent in agents:
        # if hasattr(agent, 'action_probability_map'):
        # for y in range(len(agent.action_probability_map)):
        for y in range(len(dummy_action_map)):
            #     for x in range(len(agent.action_probability_map[y])):
            for x in range(len(dummy_action_map[y])):
                position = (x, y)
                if position not in wall_positions:
                    # action_probabilities = agent.action_probability_map[y][x]
                    action_probabilities = dummy_action_map[y][x]
                    if sum(action_probabilities) > 0:  # Ensure it's not all zeros which would indicate a wall
                        # Sort actions by probability and assign colors
                        sorted_indices = sorted(range(len(action_probabilities)),
                                                key=lambda i: -action_probabilities[i])
                        colors = ['green_arrow', 'yellow_arrow', 'red_arrow', 'grey_arrow']

                        for rank, direction_index in enumerate(sorted_indices):
                            action = directions[direction_index]
                            probability = action_probabilities[direction_index]
                            arrow_color = colors[rank]
                            if probability > 0:
                                action_entity = RenderEntity(
                                    name=arrow_color,
                                    pos=position,
                                    probability=probability,
                                    rotation=direction_index * 90
                                )
                                action_entities.append(action_entity)

    renderer.render_multi_action_icons(action_entities)


def load_action_map(file_path):
    with open(file_path, 'r') as file:
        action_map = json.load(file)
    return action_map


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
