import pickle
from collections import defaultdict
from pathlib import Path
from typing import List, Dict

from stable_baselines3.common.callbacks import BaseCallback

from environments.helpers import IGNORED_DF_COLUMNS
from environments.logging.plotting import prepare_plot
import pandas as pd


class MonitorCallback(BaseCallback):

    ext = 'png'

    def __init__(self, filepath=Path('debug_out/monitor.pick'), plotting=True):
        super(MonitorCallback, self).__init__()
        self.filepath = Path(filepath)
        self._monitor_df = pd.DataFrame()
        self._monitor_dicts = defaultdict(dict)
        self.plotting = plotting
        self.started = False
        self.closed = False

    def __enter__(self):
        self._on_training_start()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._on_training_end()

    def _on_training_start(self) -> None:
        if self.started:
            pass
        else:
            self.filepath.parent.mkdir(exist_ok=True, parents=True)
            self.started = True
        pass

    def _on_training_end(self) -> None:
        if self.closed:
            pass
        else:
            # self.out_file.unlink(missing_ok=True)
            with self.filepath.open('wb') as f:
                pickle.dump(self._monitor_df.reset_index(), f, protocol=pickle.HIGHEST_PROTOCOL)
            if self.plotting:
                print('Monitor files were dumped to disk, now plotting....')

                # %% Load MonitorList from Disk
                with self.filepath.open('rb') as f:
                    monitor_list = pickle.load(f)
                df = None
                for m_idx, monitor in enumerate(monitor_list):
                    monitor['episode'] = m_idx
                    if df is None:
                        df = pd.DataFrame(columns=monitor.columns)
                    for _, row in monitor.iterrows():
                        df.loc[df.shape[0]] = row
                if df is None:  # The env exited premature, we catch it.
                    self.closed = True
                    return
                for column in list(df.columns):
                    if column != 'episode':
                        df[f'{column}_roll'] = df[column].rolling(window=50).mean()
                # result.tail()
                prepare_plot(filepath=self.filepath, results_df=df.filter(regex=(".+_roll")))
                print('Plotting done.')
            self.closed = True

    def _on_step(self, alt_infos: List[Dict] = None, alt_dones: List[bool] = None) -> bool:
        infos = alt_infos or self.locals.get('infos', [])
        if alt_dones is not None:
            dones = alt_dones
        elif self.locals.get('dones', None) is not None:
            dones =self.locals.get('dones', None)
        elif self.locals.get('dones', None) is not None:
            dones = self.locals.get('done', [None])
        else:
            dones = []

        for env_idx, (info, done) in enumerate(zip(infos, dones)):
            self._monitor_dicts[env_idx][self.num_timesteps - env_idx] = {key: val for key, val in info.items()
                                                                if key not in ['terminal_observation', 'episode']
                                                                and not key.startswith('rec_')}
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
        return True

