import pickle
from os import PathLike
from pathlib import Path
from typing import Union

from gymnasium import Wrapper

from environment.utils.helpers import IGNORED_DF_COLUMNS
from environment.factory import REC_TAC

import pandas as pd

from plotting.compare_runs import plot_single_run


class EnvMonitor(Wrapper):

    ext = 'png'

    def __init__(self, env, filepath: Union[str, PathLike] = None):
        super(EnvMonitor, self).__init__(env)
        self._filepath = filepath
        self._monitor_df = pd.DataFrame()
        self._monitor_dict = dict()

    def __getattr__(self, item):
        return getattr(self.unwrapped, item)

    def step(self, action):
        obs_type, obs, reward, done, info = self.env.step(action)
        self._read_info(info)
        self._read_done(done)
        return obs_type, obs, reward, done, info

    def reset(self):
        return self.unwrapped.reset()

    def _read_info(self, info: dict):
        self._monitor_dict[len(self._monitor_dict)] = {
            key: val for key, val in info.items() if
            key not in ['terminal_observation', 'episode'] and not key.startswith(REC_TAC)}
        return

    def _read_done(self, done):
        if done:
            env_monitor_df = pd.DataFrame.from_dict(self._monitor_dict, orient='index')
            self._monitor_dict = dict()
            columns = [col for col in env_monitor_df.columns if col not in IGNORED_DF_COLUMNS]
            env_monitor_df = env_monitor_df.aggregate(
                {col: 'mean' if col.endswith('ount') else 'sum' for col in columns}
            )
            env_monitor_df['episode'] = len(self._monitor_df)
            self._monitor_df = self._monitor_df.append([env_monitor_df])
        else:
            pass
        return

    def save_run(self, filepath: Union[Path, str, None] = None, auto_plotting_keys=None):
        filepath = Path(filepath or self._filepath)
        filepath.parent.mkdir(exist_ok=True, parents=True)
        with filepath.open('wb') as f:
            pickle.dump(self._monitor_df.reset_index(), f, protocol=pickle.HIGHEST_PROTOCOL)
        if auto_plotting_keys:
            plot_single_run(filepath, column_keys=auto_plotting_keys)
