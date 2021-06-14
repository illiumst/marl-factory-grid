import pickle
import warnings
from typing import Union, List
from os import PathLike
from pathlib import Path
import time

import pandas as pd
from gym.wrappers import FrameStack

from stable_baselines3.common.callbacks import CallbackList

from environments.factory.base_factory import MovementProperties
from environments.factory.simple_factory import DirtProperties, SimpleFactory
from environments.helpers import IGNORED_DF_COLUMNS
from environments.logging.monitor import MonitorCallback
from environments.logging.plotting import prepare_plot

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)


def combine_runs(run_path: Union[str, PathLike]):
    run_path = Path(run_path)
    df_list = list()
    for run, monitor_file in enumerate(run_path.rglob('monitor_*.pick')):
        with monitor_file.open('rb') as f:
            monitor_df = pickle.load(f)

        monitor_df['run'] = run
        monitor_df = monitor_df.fillna(0)
        df_list.append(monitor_df)

    df = pd.concat(df_list,  ignore_index=True)
    df = df.fillna(0).rename(columns={'episode': 'Episode', 'run': 'Run'})
    columns = [col for col in df.columns if col not in IGNORED_DF_COLUMNS]

    roll_n = 50
    skip_n = 40

    non_overlapp_window = df.groupby(['Run', 'Episode']).rolling(roll_n, min_periods=1).mean()

    df_melted = non_overlapp_window[columns].reset_index().melt(id_vars=['Episode', 'Run'],
                                                                value_vars=columns, var_name="Measurement",
                                                                value_name="Score")
    df_melted = df_melted[df_melted['Episode'] % skip_n == 0]

    prepare_plot(run_path / f'{run_path.name}_monitor_lineplot.png', df_melted)
    print('Plotting done.')


def compare_runs(run_path: Path, run_identifier: int, parameter: Union[str, List[str]]):
    run_path = Path(run_path)
    df_list = list()
    parameter = [parameter] if isinstance(parameter, str) else parameter
    for path in run_path.iterdir():
        if path.is_dir() and str(run_identifier) in path.name:
            for run, monitor_file in enumerate(path.rglob('monitor_*.pick')):
                with monitor_file.open('rb') as f:
                    monitor_df = pickle.load(f)

                monitor_df['run'] = run
                monitor_df['model'] = path.name.split('_')[0]
                monitor_df = monitor_df.fillna(0)
                df_list.append(monitor_df)

    df = pd.concat(df_list, ignore_index=True)
    df = df.fillna(0).rename(columns={'episode': 'Episode', 'run': 'Run', 'model': 'Model'})
    columns = [col for col in df.columns if col in parameter]

    roll_n = 40
    skip_n = 20

    non_overlapp_window = df.groupby(['Model', 'Run', 'Episode']).rolling(roll_n, min_periods=1).mean()

    df_melted = non_overlapp_window[columns].reset_index().melt(id_vars=['Episode', 'Run', 'Model'],
                                                                value_vars=columns, var_name="Measurement",
                                                                value_name="Score")
    df_melted = df_melted[df_melted['Episode'] % skip_n == 0]

    style = 'Measurement' if len(columns) > 1 else None
    prepare_plot(run_path / f'{run_identifier}_compare_{parameter}.png', df_melted, hue='Model', style=style)
    print('Plotting done.')


if __name__ == '__main__':

    # compare_runs(Path('debug_out'), 1623052687, ['step_reward'])
    # exit()

    from stable_baselines3 import PPO, DQN, A2C
    from algorithms.reg_dqn import RegDQN
    # from sb3_contrib import QRDQN

    dirt_props = DirtProperties(clean_amount=3, gain_amount=0.2, max_global_amount=30,
                                max_local_amount=5, spawn_frequency=1, max_spawn_ratio=0.05)
    move_props = MovementProperties(allow_diagonal_movement=True,
                                    allow_square_movement=True,
                                    allow_no_op=False)
    time_stamp = int(time.time())

    out_path = None

    for modeL_type in [A2C, PPO, RegDQN, DQN]:  # , QRDQN]:
        for seed in range(3):

            with SimpleFactory(n_agents=1, dirt_properties=dirt_props, pomdp_radius=2, max_steps=400,
                               movement_properties=move_props, level_name='rooms', frames_to_stack=4,
                               omit_agent_slice_in_obs=False, combin_agent_slices_in_obs=True) as env:

                if modeL_type.__name__ in ["PPO", "A2C"]:
                    kwargs = dict(ent_coef=0.01)
                elif modeL_type.__name__ in ["RegDQN", "DQN", "QRDQN"]:
                    kwargs = dict(target_update_interval=500, buffer_size=30000, learning_starts=5000,
                                  exploration_final_eps=0.01, batch_size=96)
                else:
                    raise NameError(f'The model "{model.__name__}" has the wrong name.')
                model = modeL_type("MlpPolicy", env, verbose=1, seed=seed, device='cpu', **kwargs)

                out_path = Path('debug_out') / f'{model.__class__.__name__}_{time_stamp}'

                # identifier = f'{seed}_{model.__class__.__name__}_{time_stamp}'
                identifier = f'{seed}_{model.__class__.__name__}_{time_stamp}'
                out_path /= identifier

                callbacks = CallbackList(
                    [MonitorCallback(filepath=out_path / f'monitor_{identifier}.pick', plotting=False)]
                )

                model.learn(total_timesteps=int(1e5), callback=callbacks)

                save_path = out_path / f'model_{identifier}.zip'
                save_path.parent.mkdir(parents=True, exist_ok=True)
                model.save(save_path)
                env.save_params(out_path.parent / f'env_{model.__class__.__name__}_{time_stamp}.yaml')

        if out_path:
            combine_runs(out_path.parent)
    if out_path:
        compare_runs(Path('debug_out'), time_stamp, 'step_reward')
