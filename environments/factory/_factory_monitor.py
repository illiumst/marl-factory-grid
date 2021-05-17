from collections import defaultdict


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
        df.loc[0] = df.iloc[0].fillna(0)
        df = df.fillna(method='ffill')
        return df

    def reset(self):
        raise RuntimeError("DO NOT DO THIS! Always initalize a new Monitor per Env-Run.")