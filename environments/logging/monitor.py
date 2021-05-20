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
        return df

    def reset(self):
        raise RuntimeError("DO NOT DO THIS! Always initalize a new Monitor per Env-Run.")


class MonitorCallback(BaseCallback):

    ext = 'png'

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
            self.prepare_plot()
            self.closed = True

    def _on_step(self) -> bool:
        if self.locals['dones'].item():
            self._monitor_list.append(self.env.monitor)
        else:
            pass

    def plot(self, **kwargs):
        from matplotlib import pyplot as plt
        plt.rcParams.update(kwargs)

        plt.tight_layout()
        figure = plt.gcf()
        plt.show()
        figure.savefig(str(self.filepath.parent / f'{self.filepath.stem}_monitor_measures.{self.ext}'), format=self.ext)

    def prepare_plot(self):
        # %% Imports
        import pandas as pd
        import seaborn as sns

        # %% Load MonitorList from Disk
        with self.filepath.open('rb') as f:
            monitor_list = pickle.load(f)

        result = pd.concat(monitor_list, sort=False)
        # result.tail()

        # %%
        lineplot = sns.lineplot(data=result)
        lineplot.title.title = f'Lineplot Summary of {len(monitor_list)} Episodes'

        # %%
        sns.set_theme(palette='husl', style='whitegrid')
        font_size = 16
        tex_fonts = {
            # Use LaTeX to write all text
            "text.usetex": True,
            "font.family": "serif",
            # Use 10pt font in plots, to match 10pt font in document
            "axes.labelsize": font_size,
            "font.size": font_size,
            # Make the legend/label fonts a little smaller
            "legend.fontsize": font_size - 2,
            "xtick.labelsize": font_size - 2,
            "ytick.labelsize": font_size - 2
        }

        try:
            self.plot(**tex_fonts)
        except FileNotFoundError:
            tex_fonts['text.usetex'] = False
            self.plot(**tex_fonts)
