import pickle
from os import PathLike
from pathlib import Path
from typing import Union

import pandas as pd

from marl_factory_grid.utils.helpers import IGNORED_DF_COLUMNS
from marl_factory_grid.utils.plotting.plotting_utils import prepare_plot


def plot_single_run(run_path: Union[str, PathLike], use_tex: bool = False, column_keys=None,
                    file_key: str ='monitor', file_ext: str ='pkl'):
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

    df = pd.concat(df_list,  ignore_index=True)
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
