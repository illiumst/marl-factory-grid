import pickle
from pathlib import Path
from collections import defaultdict

from stable_baselines3.common.callbacks import BaseCallback


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
        try:
            df.loc[0] = df.iloc[0].fillna(0)
        except IndexError:
            return None
        df = df.fillna(method='ffill')
        return

    def reset(self):
        raise RuntimeError("DO NOT DO THIS! Always initalize a new Monitor per Env-Run.")


class MonitorCallback(BaseCallback):

    def __init__(self, env, filepath=Path('debug_out/monitor.pick')):
        super(MonitorCallback, self).__init__()
        self.filepath = Path(filepath)
        self._monitor_list = list()
        self.env = env
        self.started = False
        self.closed = False

    @property
    def monitor_as_df_list(self):
        return [x.to_pd_dataframe() for x in self._monitor_list]

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
                pickle.dump(self.monitor_as_df_list, f, protocol=pickle.HIGHEST_PROTOCOL)
            self.closed = True

    def _on_step(self) -> bool:
        if self.locals['dones'].item():
            self._monitor_list.append(self.env.monitor)
        else:
            pass
