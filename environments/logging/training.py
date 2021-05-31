from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from stable_baselines3.common.callbacks import BaseCallback

from environments.logging.plotting import prepare_plot


class TraningMonitor(BaseCallback):

    def __init__(self, filepath, flush_interval=None):
        super(TraningMonitor, self).__init__()
        self.values = defaultdict(dict)
        self.rewards = defaultdict(lambda: 0)

        self.filepath = Path(filepath)
        self.flush_interval = flush_interval
        self.next_flush: int
        pass

    def _on_training_start(self) -> None:
        self.flush_interval = self.flush_interval or (self.locals['total_timesteps'] * 0.1)
        self.next_flush = self.flush_interval

    def _flush(self):
        df = pd.DataFrame.from_dict(self.values,  orient='index')
        if not self.filepath.exists():
            df.to_csv(self.filepath, mode='wb', header=True)
        else:
            df.to_csv(self.filepath, mode='a', header=False)

    def _on_step(self) -> bool:
        for idx, done in np.ndenumerate(self.locals.get('dones', [])):
            idx = idx[0]
            # self.values[self.num_timesteps].update(**{f'reward_env_{idx}': self.locals['rewards'][idx]})
            self.rewards[idx] += self.locals['rewards'][idx]
            if done:
                self.values[self.num_timesteps].update(**{f'acc_epispde_r_env_{idx}': self.rewards[idx]})
                self.rewards[idx] = 0

        if self.num_timesteps >= self.next_flush and self.values:
            self._flush()
            self.values = defaultdict(dict)

        self.next_flush += self.flush_interval
        return True

    def on_training_end(self) -> None:
        self._flush()
        self.values = defaultdict(dict)
        # prepare_plot()

