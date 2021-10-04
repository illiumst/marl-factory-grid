import pickle
from collections import defaultdict
from pathlib import Path
from typing import List, Dict

from stable_baselines3.common.callbacks import BaseCallback

from environments.helpers import IGNORED_DF_COLUMNS

import pandas as pd


class MonitorCallback(BaseCallback):

    ext = 'png'

    def __init__(self, filepath=Path('debug_out/monitor.pick')):
        super(MonitorCallback, self).__init__()
        self.filepath = Path(filepath)
        self._monitor_df = pd.DataFrame()
        self._monitor_dicts = defaultdict(dict)
        self.started = False
        self.closed = False

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    def _on_training_start(self) -> None:
        if self.started:
            pass
        else:
            self.start()
        pass

    def _on_training_end(self) -> None:
        if self.closed:
            pass
        else:
            self.stop()

    def _on_step(self, alt_infos: List[Dict] = None, alt_dones: List[bool] = None) -> bool:
        if self.started:
            for env_idx, info in enumerate(self.locals.get('infos', [])):
                self.read_info(env_idx, info)

            for env_idx, done in list(
                    enumerate(self.locals.get('dones', []))) + list(enumerate(self.locals.get('done', []))):
                self.read_done(env_idx, done)
        else:
            pass
        return True

    def read_info(self, env_idx, info: dict):
        self._monitor_dicts[env_idx][len(self._monitor_dicts[env_idx])] = {
            key: val for key, val in info.items() if
            key not in ['terminal_observation', 'episode'] and not key.startswith('rec_')}
        return

    def read_done(self, env_idx, done):
        if done:
            env_monitor_df = pd.DataFrame.from_dict(self._monitor_dicts[env_idx], orient='index')
            self._monitor_dicts[env_idx] = dict()
            columns = [col for col in env_monitor_df.columns if col not in IGNORED_DF_COLUMNS]
            env_monitor_df = env_monitor_df.aggregate(
                {col: 'mean' if col.endswith('ount') else 'sum' for col in columns}
            )
            env_monitor_df['episode'] = len(self._monitor_df)
            self._monitor_df = self._monitor_df.append([env_monitor_df])
        else:
            pass
        return

    def stop(self):
        # self.out_file.unlink(missing_ok=True)
        with self.filepath.open('wb') as f:
            pickle.dump(self._monitor_df.reset_index(), f, protocol=pickle.HIGHEST_PROTOCOL)
        self.closed = True

    def start(self):
        if self.started:
            pass
        else:
            self.filepath.parent.mkdir(exist_ok=True, parents=True)
            self.started = True
        pass
