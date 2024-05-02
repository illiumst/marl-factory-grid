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


rotation_mapping = {
    'north': ('cardinal', 0),
    'east': ('cardinal', 270),
    'south': ('cardinal', 180),
    'west': ('cardinal', 90),
    'northeast': ('diagonal', 0),
    'southeast': ('diagonal', 270),
    'southwest': ('diagonal', 180),
    'northwest': ('diagonal', 90)
}


def plot_routes(factory, agents):
    renderer = Renderer(factory.map.level_shape, custom_assets_path={
        'cardinal': 'marl_factory_grid/utils/plotting/action_assets/cardinal.png',
        'diagonal': 'marl_factory_grid/utils/plotting/action_assets/diagonal.png',
        'door': 'marl_factory_grid/utils/plotting/action_assets/door.png',
        'wall': 'marl_factory_grid/environment/assets/wall.png'})

    wall_positions = factory.map.walls

    for index, agent in enumerate(agents):
        action_entities = []
        # Add walls to the action_entities list
        for pos in wall_positions:
            wall_entity = RenderEntity(
                name='wall',
                probability=1.0,
                pos=np.array(pos),
            )
            action_entities.append(wall_entity)
        current_position = agent.spawn_position

        if hasattr(agent, 'action_probabilities'):
            # Handle RL agents with action probabilities
            top_actions = sorted(agent.action_probabilities.items(), key=lambda x: -x[1])[:4]
        else:
            # Handle deterministic agents by iterating through all actions in the list
            top_actions = [(action, 1.0) for action in agent.action_list]

        for action, probability in top_actions:
            base_icon, rotation = rotation_mapping.get(action.lower(), ('north', 0))
            icon_name = base_icon
            new_position = action_to_coords(current_position, action.lower())
            print(f"current position type and value: {type(current_position)}, {new_position}")

            action_entity = RenderEntity(
                name=icon_name,
                pos=np.array(current_position),
                probability=probability,
                rotation=rotation
            )
            action_entities.append(action_entity)
            current_position = new_position

        renderer.render_action_icons(action_entities)


def action_to_coords(current_position, action):
    direction_mapping = {
        'north': (0, -1),
        'south': (0, 1),
        'east': (1, 0),
        'west': (-1, 0),
        'northeast': (1, -1),
        'northwest': (-1, -1),
        'southeast': (1, 1),
        'southwest': (-1, 1)
    }
    delta = direction_mapping.get(action)
    if delta is not None:
        new_position = [current_position[0] + delta[0], current_position[1] + delta[1]]
        return new_position
    print(f"No valid action found for {action}.")
    return current_position

