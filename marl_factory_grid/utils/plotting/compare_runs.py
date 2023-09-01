import pickle
import re
from os import PathLike
from pathlib import Path
from typing import Union, List

import pandas as pd

from marl_factory_grid.utils.helpers import IGNORED_DF_COLUMNS
from marl_factory_grid.utils.plotting.plotting import prepare_plot

MODEL_MAP = None


def plot_single_run(run_path: Union[str, PathLike], use_tex: bool = False, column_keys=None):
    run_path = Path(run_path)
    df_list = list()
    if run_path.is_dir():
        monitor_file = next(run_path.glob('*monitor*.pick'))
    elif run_path.exists() and run_path.is_file():
        monitor_file = run_path
    else:
        raise ValueError

    with monitor_file.open('rb') as f:
        monitor_df = pickle.load(f)

        monitor_df = monitor_df.fillna(0)
        df_list.append(monitor_df)

    df = pd.concat(df_list,  ignore_index=True)
    df = df.fillna(0).rename(columns={'episode': 'Episode'}).sort_values(['Episode'])
    if column_keys is not None:
        columns = [col for col in column_keys if col in df.columns]
    else:
        columns = [col for col in df.columns if col not in IGNORED_DF_COLUMNS]

    roll_n = 50

    non_overlapp_window = df.groupby(['Episode']).rolling(roll_n, min_periods=1).mean()

    df_melted = df[columns + ['Episode']].reset_index().melt(
        id_vars=['Episode'], value_vars=columns, var_name="Measurement", value_name="Score"
    )

    if df_melted['Episode'].max() > 800:
        skip_n = round(df_melted['Episode'].max() * 0.02)
        df_melted = df_melted[df_melted['Episode'] % skip_n == 0]

    prepare_plot(run_path.parent / f'{run_path.parent.name}_monitor_lineplot.png', df_melted, use_tex=use_tex)
    print('Plotting done.')


def compare_seed_runs(run_path: Union[str, PathLike], use_tex: bool = False):
    run_path = Path(run_path)
    df_list = list()
    for run, monitor_file in enumerate(run_path.rglob('monitor*.pick')):
        with monitor_file.open('rb') as f:
            monitor_df = pickle.load(f)

        monitor_df['run'] = run
        monitor_df = monitor_df.fillna(0)
        df_list.append(monitor_df)

    df = pd.concat(df_list,  ignore_index=True)
    df = df.fillna(0).rename(columns={'episode': 'Episode', 'run': 'Run'}).sort_values(['Run', 'Episode'])
    columns = [col for col in df.columns if col not in IGNORED_DF_COLUMNS]

    roll_n = 50

    non_overlapp_window = df.groupby(['Run', 'Episode']).rolling(roll_n, min_periods=1).mean()

    df_melted = non_overlapp_window[columns].reset_index().melt(id_vars=['Episode', 'Run'],
                                                                value_vars=columns, var_name="Measurement",
                                                                value_name="Score")

    if df_melted['Episode'].max() > 800:
        skip_n = round(df_melted['Episode'].max() * 0.02)
        df_melted = df_melted[df_melted['Episode'] % skip_n == 0]

    run_path.mkdir(parents=True, exist_ok=True)
    if run_path.exists() and run_path.is_file():
        prepare_plot(run_path.parent / f'{run_path.name}_monitor_lineplot.png', df_melted, use_tex=use_tex)
    else:
        prepare_plot(run_path / f'{run_path.name}_monitor_lineplot.png', df_melted, use_tex=use_tex)
    print('Plotting done.')


def compare_model_runs(run_path: Path, run_identifier: Union[str, int], parameter: Union[str, List[str]],
                       use_tex: bool = False):
    run_path = Path(run_path)
    df_list = list()
    parameter = [parameter] if isinstance(parameter, str) else parameter
    for path in run_path.iterdir():
        if path.is_dir() and str(run_identifier) in path.name:
            for run, monitor_file in enumerate(path.rglob('monitor*.pick')):
                with monitor_file.open('rb') as f:
                    monitor_df = pickle.load(f)

                monitor_df['run'] = run
                monitor_df['model'] = next((x for x in path.name.split('_') if x in MODEL_MAP.keys()))
                monitor_df = monitor_df.fillna(0)
                df_list.append(monitor_df)

    df = pd.concat(df_list, ignore_index=True)
    df = df.fillna(0).rename(columns={'episode': 'Episode', 'run': 'Run', 'model': 'Model'})
    columns = [col for col in df.columns if col in parameter]

    last_episode_to_report = min(df.groupby(['Model'])['Episode'].max())
    df = df[df['Episode'] < last_episode_to_report]

    roll_n = 40
    non_overlapp_window = df.groupby(['Model', 'Run', 'Episode']).rolling(roll_n, min_periods=1).mean()

    df_melted = non_overlapp_window[columns].reset_index().melt(id_vars=['Episode', 'Run', 'Model'],
                                                                value_vars=columns, var_name="Measurement",
                                                                value_name="Score")

    if df_melted['Episode'].max() > 80:
        skip_n = round(df_melted['Episode'].max() * 0.02)
        df_melted = df_melted[df_melted['Episode'] % skip_n == 0]

    style = 'Measurement' if len(columns) > 1 else None
    prepare_plot(run_path / f'{run_identifier}_compare_{parameter}.png', df_melted, hue='Model', style=style,
                 use_tex=use_tex)
    print('Plotting done.')


def compare_all_parameter_runs(run_root_path: Path, parameter: Union[str, List[str]],
                               param_names: Union[List[str], None] = None, str_to_ignore='', use_tex: bool = False):
    run_root_path = Path(run_root_path)
    df_list = list()
    parameter = [parameter] if isinstance(parameter, str) else parameter
    for monitor_idx, monitor_file in enumerate(run_root_path.rglob('monitor*.pick')):
        with monitor_file.open('rb') as f:
            monitor_df = pickle.load(f)

        params = [x.name for x in monitor_file.parents if x.parent not in run_root_path.parents]
        if str_to_ignore:
            params = [re.sub(f'_*({str_to_ignore})', '', param) for param in params]

        if monitor_idx == 0:
            if param_names is not None:
                if len(param_names) < len(params):
                    # FIXME: Missing Seed Detection, see below @111
                    param_names = [next(param_names) if param not in MODEL_MAP.keys() else 'Model' for param in params]
                elif len(param_names) == len(params):
                    pass
                else:
                    raise ValueError
            else:
                param_names = []
        for param_idx, param in enumerate(params):
            dtype = None
            if param in MODEL_MAP.keys():
                param_name = 'Model'
            elif '_' in param:
                param_split = param.split('_')
                if len(param_split) == 2 and any(split in MODEL_MAP.keys() for split in param_split):
                    # Extract the seed
                    param = int(next(x for x in param_split if x not in MODEL_MAP))
                    param_name = 'Seed'
                    dtype = int
                else:
                    param_name = f'param_{param_idx}'
            else:
                param_name = f'param_{param_idx}'
            dtype = dtype if dtype is not None else str
            monitor_df[param_name] = str(param)
            monitor_df[param_name] = monitor_df[param_name].astype(dtype)
            if monitor_idx == 0:
                param_names.append(param_name)

        monitor_df = monitor_df.fillna(0)
        df_list.append(monitor_df)

    df = pd.concat(df_list, ignore_index=True)
    df = df.fillna(0).rename(columns={'episode': 'Episode'}).sort_values(['Episode'])

    for param_name in param_names:
        df[param_name] = df[param_name].astype(str)
    columns = [col for col in df.columns if col in parameter]

    last_episode_to_report = min(df.groupby(['Model'])['Episode'].max())
    df = df[df['Episode'] < last_episode_to_report]

    if df['Episode'].max() > 80:
        skip_n = round(df['Episode'].max() * 0.02)
        df = df[df['Episode'] % skip_n == 0]
    combinations = [x for x in param_names if x not in ['Model', 'Seed']]
    df['Parameter Combination'] = df[combinations].apply(lambda row: '_'.join(row.values.astype(str)), axis=1)
    df.drop(columns=combinations, inplace=True)

    # non_overlapp_window = df.groupby(param_names).sum()

    df_melted = df.reset_index().melt(id_vars=['Parameter Combination', 'Episode'],
                                      value_vars=columns, var_name="Measurement",
                                      value_name="Score")

    style = 'Measurement' if len(columns) > 1 else None
    prepare_plot(run_root_path / f'compare_{parameter}.png', df_melted, hue='Parameter Combination',
                 style=style, use_tex=use_tex)
    print('Plotting done.')
