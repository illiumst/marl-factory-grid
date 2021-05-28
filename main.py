import pickle
import warnings
from typing import Union
from os import PathLike
from pathlib import Path
import time
import pandas as pd

from stable_baselines3.common.callbacks import CallbackList

from environments.factory.simple_factory import DirtProperties, SimpleFactory
from environments.logging.monitor import MonitorCallback
from environments.logging.plotting import prepare_plot
from environments.logging.training import TraningMonitor

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)


def combine_runs(run_path: Union[str, PathLike]):
    run_path = Path(run_path)
    df_list = list()
    for run, monitor_file in enumerate(run_path.rglob('monitor_*.pick')):
        with monitor_file.open('rb') as f:
            monitor_list = pickle.load(f)

        for m_idx in range(len(monitor_list)):
            monitor_list[m_idx]['episode'] = m_idx
            monitor_list[m_idx]['run'] = run

        df = pd.concat(monitor_list, ignore_index=True)
        df['train_step'] = range(df.shape[0])

        df = df.fillna(0)

        #for column in list(df.columns):
        #    if column not in ['episode', 'run', 'step', 'train_step']:
        #        if 'clean' in column or '_vs_' in column:
        #            df[f'{column}_sum_roll'] = df[column].rolling(window=50, min_periods=1).sum()
        #        else:
        #            df[f'{column}_mean_roll'] = df[column].rolling(window=50, min_periods=1).mean()

        df_list.append(df)
    df = pd.concat(df_list,  ignore_index=True)
    df = df.fillna(0).rename(columns={'episode': 'Episode', 'run': 'Run'})
    columns = [col for col in df.columns if col not in ['Episode', 'Run', 'train_step', 'step']]

    df_group = df.groupby(['Episode', 'Run']).aggregate(
        {col: 'mean' if col in ['dirt_amount', 'dirty_tiles'] else 'sum' for col in columns}
    )

    non_overlapp_window = df_group.groupby(['Run', (df_group.index.get_level_values('Episode') // 20)]).mean()

    df_melted = non_overlapp_window.reset_index().melt(id_vars=['Episode', 'Run'],
                                                       value_vars=columns, var_name="Measurement",
                                                       value_name="Score")

    prepare_plot(run_path / f'{run_path.name}_monitor_lineplot.png', df_melted)
    print('Plotting done.')


if __name__ == '__main__':

    combine_runs('debug_out/PPO_1622120377')
    exit()

    from stable_baselines3 import PPO  # DQN
    dirt_props = DirtProperties()
    time_stamp = int(time.time())

    out_path = None

    for seed in range(5):

        env = SimpleFactory(n_agents=1, dirt_properties=dirt_props, allow_diagonal_movement=False, allow_no_op=False)

        model = PPO("MlpPolicy", env, verbose=1, ent_coef=0.0, seed=seed, device='cpu')

        out_path = Path('debug_out') / f'{model.__class__.__name__}_{time_stamp}'

        identifier = f'{seed}_{model.__class__.__name__}_{time_stamp}'
        out_path /= identifier

        callbacks = CallbackList(
            [TraningMonitor(out_path / f'train_logging_{identifier}.csv'),
             MonitorCallback(env, filepath=out_path / f'monitor_{identifier}.pick', plotting=False)]
        )

        model.learn(total_timesteps=int(2e6), callback=callbacks)

        save_path = out_path / f'model_{identifier}.zip'
        save_path.parent.mkdir(parents=True, exist_ok=True)
        model.save(save_path)

    if out_path:
        combine_runs(out_path)
