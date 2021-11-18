import json
from collections import defaultdict
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
import simplejson
from stable_baselines3.common.callbacks import BaseCallback

from environments.factory.base.base_factory import REC_TAC


# noinspection PyAttributeOutsideInit
from environments.helpers import Constants as c


class RecorderCallback(BaseCallback):

    def __init__(self, filepath: Union[str, Path], occupation_map: bool = False, trajectory_map: bool = False,
                 entities='all'):
        super(RecorderCallback, self).__init__()
        self.trajectory_map = trajectory_map
        self.occupation_map = occupation_map
        self.filepath = Path(filepath)
        self._recorder_dict = defaultdict(list)
        self._recorder_out_list = list()
        self._env_params = None
        self.do_record: bool
        if isinstance(entities, str):
            if entities.lower() == 'all':
                self._entities = None
            else:
                self._entities = [entities]
        else:
            self._entities = entities
        self.started = False
        self.closed = False

    def read_params(self, params):
        self._env_params = params

    def read_info(self, env_idx, info: dict):
        if info_dict := {key.replace(REC_TAC, ''): val for key, val in info.items() if key.startswith(f'{REC_TAC}')}:
            if self._entities:
                info_dict = {k: v for k, v in info_dict.items() if k in self._entities}

            info_dict.update(episode=(self.num_timesteps + env_idx))
            self._recorder_dict[env_idx].append(info_dict)
        else:
            pass
        return

    def read_done(self, env_idx, done):
        if done:
            self._recorder_out_list.append({'steps': self._recorder_dict[env_idx],
                                            'episode': len(self._recorder_out_list)})
            self._recorder_dict[env_idx] = list()
        else:
            pass

    def start(self, force=False):
        if (hasattr(self.training_env, 'record_episodes') and self.training_env.record_episodes) or force:
            self.do_record = True
            self.filepath.parent.mkdir(exist_ok=True, parents=True)
            self.started = True
        else:
            self.do_record = False

    def stop(self):
        if self.do_record and self.started:
            # self.out_file.unlink(missing_ok=True)
            with self.filepath.open('w') as f:
                out_dict = {'episodes': self._recorder_out_list, 'header': self._env_params}
                try:
                    simplejson.dump(out_dict, f, indent=4)
                except TypeError:
                    print('Shit')

            if self.occupation_map:
                a = np.zeros((15, 15))
                for episode in  out_dict['episodes']:
                    df = pd.DataFrame([y for x in episode['steps'] for y in x['Agents']])

                    b = list(df[['x', 'y']].to_records(index=False))

                    np.add.at(a, tuple(zip(*b)), 1)

                # a = np.rot90(a)
                import seaborn as sns
                from matplotlib import pyplot as plt
                hm = sns.heatmap(data=a)
                hm.set_title('Very Nice Heatmap')
                plt.show()

            if self.trajectory_map:
                print('Recorder files were dumped to disk, now plotting the occupation map...')

            self.closed = True
            self.started = False
        else:
            pass

    def _on_step(self) -> bool:
        if self.do_record and self.started:
            for env_idx, info in enumerate(self.locals.get('infos', [])):
                self.read_info(env_idx, info)

            for env_idx, done in list(
                    enumerate(self.locals.get('dones', []))) + list(
                enumerate(self.locals.get('done', []))):
                self.read_done(env_idx, done)
        else:
            pass
        return True

    def __enter__(self):
        self.start(force=True)
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
