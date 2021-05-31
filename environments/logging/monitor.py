import pickle
from pathlib import Path
from collections import defaultdict

from stable_baselines3.common.callbacks import BaseCallback

from environments.helpers import IGNORED_DF_COLUMNS
from environments.logging.plotting import prepare_plot
import pandas as pd


class FactoryMonitor:

    def __init__(self, env):
        self._env = env
        self._monitor = defaultdict(lambda: defaultdict(lambda: 0))
        self._last_vals = defaultdict(lambda: 0)

    def __iter__(self):
        for key, value in self._monitor.items():
            yield key, dict(value)

    def add(self, key, value, step=None):
        assert step is None or step >= 1                                            # Is this good practice?
        step = step or self._env.steps
        self._last_vals[key] = self._last_vals[key] + value
        self._monitor[key][step] = self._last_vals[key]
        return self._last_vals[key]

    def set(self, key, value, step=None):
        assert step is None or step >= 1                                            # Is this good practice?
        step = step or self._env.steps
        self._last_vals[key] = value
        self._monitor[key][step] = self._last_vals[key]
        return self._last_vals[key]

    def remove(self, key, value, step=None):
        assert step is None or step >= 1                                            # Is this good practice?
        step = step or self._env.steps
        self._last_vals[key] = self._last_vals[key] - value
        self._monitor[key][step] = self._last_vals[key]
        return self._last_vals[key]

    def to_dict(self):
        return dict(self)

    def to_pd_dataframe(self):
        import pandas as pd
        df = pd.DataFrame.from_dict(self.to_dict())
        df.fillna(0)
        return df

    def reset(self):
        raise RuntimeError("DO NOT DO THIS! Always initalize a new Monitor per Env-Run.")


class MonitorCallback(BaseCallback):

    ext = 'png'

    def __init__(self, env, filepath=Path('debug_out/monitor.pick'), plotting=True):
        super(MonitorCallback, self).__init__()
        self.filepath = Path(filepath)
        self._monitor_df = pd.DataFrame()
        self.env = env
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

    def _on_step(self) -> bool:
        for env_idx, done in enumerate(self.locals.get('dones', [])):
            if done:
                env_monitor_df = self.locals['infos'][env_idx]['monitor'].to_pd_dataframe()
                columns = [col for col in env_monitor_df.columns if col not in IGNORED_DF_COLUMNS]
                env_monitor_df = env_monitor_df.aggregate(
                    {col: 'mean' if 'amount' in col or 'count' in col else 'sum' for col in columns}
                )
                env_monitor_df['episode'] = len(self._monitor_df)
                self._monitor_df = self._monitor_df.append([env_monitor_df])
            else:
                pass
        return True

