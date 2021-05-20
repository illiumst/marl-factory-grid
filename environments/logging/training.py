from pathlib import Path

import pandas as pd
from stable_baselines3.common.callbacks import BaseCallback


class TraningMonitor(BaseCallback):

    def __init__(self, filepath, flush_interval=None):
        super(TraningMonitor, self).__init__()
        self.values = dict()
        self.filepath = Path(filepath)
        self.flush_interval = flush_interval
        pass

    def _on_training_start(self) -> None:
        self.flush_interval = self.flush_interval or (self.locals['total_timesteps'] * 0.1)

    def _flush(self):
        df = pd.DataFrame.from_dict(self.values)
        if not self.filepath.exists():
            df.to_csv(self.filepath, mode='wb', header=True)
        else:
            df.to_csv(self.filepath, mode='a', header=False)
        self.values = dict()

    def _on_step(self) -> bool:
        self.values[self.num_timesteps] = dict(reward=self.locals['rewards'].item())
        if self.num_timesteps % self.flush_interval == 0:
            self._flush()
        return True

    def on_training_end(self) -> None:
        self._flush()

