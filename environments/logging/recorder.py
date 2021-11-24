from collections import defaultdict
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
import simplejson
from stable_baselines3.common.callbacks import BaseCallback

from environments.factory.base.base_factory import REC_TAC


class EnvRecorder(BaseCallback):

    def __init__(self, env, entities='all'):
        super(EnvRecorder, self).__init__()
        self.unwrapped = env
        self._recorder_dict = defaultdict(list)
        self._recorder_out_list = list()
        if isinstance(entities, str):
            if entities.lower() == 'all':
                self._entities = None
            else:
                self._entities = [entities]
        else:
            self._entities = entities
        self.started = False
        self.closed = False

    def __getattr__(self, item):
        return getattr(self.unwrapped, item)

    def reset(self):
        self.unwrapped._record_episodes = True
        return self.unwrapped.reset()

    def _on_training_start(self) -> None:
        self.unwrapped._record_episodes = True
        pass

    def _read_info(self, env_idx, info: dict):
        if info_dict := {key.replace(REC_TAC, ''): val for key, val in info.items() if key.startswith(f'{REC_TAC}')}:
            if self._entities:
                info_dict = {k: v for k, v in info_dict.items() if k in self._entities}

            info_dict.update(episode=(self.num_timesteps + env_idx))
            self._recorder_dict[env_idx].append(info_dict)
        else:
            pass
        return

    def _read_done(self, env_idx, done):
        if done:
            self._recorder_out_list.append({'steps': self._recorder_dict[env_idx],
                                            'episode': len(self._recorder_out_list)})
            self._recorder_dict[env_idx] = list()
        else:
            pass

    def save_records(self, filepath: Union[Path, str], save_occupation_map=False, save_trajectory_map=False):
        filepath = Path(filepath)
        filepath.parent.mkdir(exist_ok=True, parents=True)
        # self.out_file.unlink(missing_ok=True)
        with filepath.open('w') as f:
            out_dict = {'episodes': self._recorder_out_list, 'header': self.unwrapped.params}
            try:
                simplejson.dump(out_dict, f, indent=4)
            except TypeError:
                print('Shit')

        if save_occupation_map:
            a = np.zeros((15, 15))
            for episode in out_dict['episodes']:
                df = pd.DataFrame([y for x in episode['steps'] for y in x['Agents']])

                b = list(df[['x', 'y']].to_records(index=False))

                np.add.at(a, tuple(zip(*b)), 1)

            # a = np.rot90(a)
            import seaborn as sns
            from matplotlib import pyplot as plt
            hm = sns.heatmap(data=a)
            hm.set_title('Very Nice Heatmap')
            plt.show()

        if save_trajectory_map:
            raise NotImplementedError('This has not yet been implemented.')

    def _on_step(self) -> bool:
        for env_idx, info in enumerate(self.locals.get('infos', [])):
            self._read_info(env_idx, info)

        dones = list(enumerate(self.locals.get('dones', [])))
        dones.extend(list(enumerate(self.locals.get('done', []))))
        for env_idx, done in dones:
            self._read_done(env_idx, done)

        return True

    def _on_training_end(self) -> None:
        pass
