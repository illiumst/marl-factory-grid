import warnings
from collections import defaultdict
from os import PathLike
from pathlib import Path
from typing import Union

from gymnasium import Wrapper

import numpy as np
import pandas as pd
import simplejson

from environment.factory import REC_TAC


class EnvRecorder(Wrapper):

    def __init__(self, env, entities: str = 'all', filepath: Union[str, PathLike] = None, freq: int = 0):
        super(EnvRecorder, self).__init__(env)
        self.filepath = filepath
        self.freq = freq
        self._recorder_dict = defaultdict(list)
        self._recorder_out_list = list()
        self._episode_counter = 1
        self._do_record_dict = defaultdict(lambda: False)
        if isinstance(entities, str):
            if entities.lower() == 'all':
                self._entities = None
            else:
                self._entities = [entities]
        else:
            self._entities = entities

    def __getattr__(self, item):
        return getattr(self.unwrapped, item)

    def reset(self):
        self._on_training_start()
        return self.unwrapped.reset()

    def _on_training_start(self) -> None:
        assert self.start_recording()

    def _read_info(self, env_idx, info: dict):
        if info_dict := {key.replace(REC_TAC, ''): val for key, val in info.items() if key.startswith(f'{REC_TAC}')}:
            if self._entities:
                info_dict = {k: v for k, v in info_dict.items() if k in self._entities}
            self._recorder_dict[env_idx].append(info_dict)
        else:
            pass
        return True

    def _read_done(self, env_idx, done):
        if done:
            self._recorder_out_list.append({'steps': self._recorder_dict[env_idx],
                                            'episode': self._episode_counter})
            self._recorder_dict[env_idx] = list()
        else:
            pass

    def step(self, actions):
        step_result = self.unwrapped.step(actions)
        if self.do_record_episode(0):
            info = step_result[-1]
            self._read_info(0, info)
        if self._do_record_dict[0]:
            self._read_done(0, step_result[-2])
        return step_result

    def finalize(self):
        self._on_training_end()
        return True

    def save_records(self, filepath: Union[Path, str, None] = None,
                     only_deltas=True,
                     save_occupation_map=False,
                     save_trajectory_map=False,
                     ):
        filepath = Path(filepath or self.filepath)
        filepath.parent.mkdir(exist_ok=True, parents=True)
        # cls.out_file.unlink(missing_ok=True)
        with filepath.open('w') as f:
            if only_deltas:
                from deepdiff import DeepDiff, Delta
                diff_dict = [DeepDiff(t1,t2, ignore_order=True)
                             for t1, t2 in zip(self._recorder_out_list, self._recorder_out_list[1:])
                             ]
                out_dict = {'episodes': diff_dict}

            else:
                out_dict = {'episodes': self._recorder_out_list}
            out_dict.update(
                {'n_episodes': self._episode_counter,
                 'env_params': self.env.params,
                 'header': self.env.summarize_header
                 })
            try:
                simplejson.dump(out_dict, f, indent=4)
            except TypeError:
                print('Shit')

        if save_occupation_map:
            a = np.zeros((15, 15))
            # noinspection PyTypeChecker
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

    def do_record_episode(self, env_idx):
        if not self._recorder_dict[env_idx]:
            if self.freq:
                self._do_record_dict[env_idx] = (self.freq == -1) or (self._episode_counter % self.freq) == 0
            else:
                self._do_record_dict[env_idx] = False
                warnings.warn('You did wrap your Environment with a recorder, but set the freq to zero\n'
                              'Nothing will be recorded')
            self._episode_counter += 1
        else:
            pass
        return self._do_record_dict[env_idx]

    def _on_step(self) -> bool:
        for env_idx, info in enumerate(self.locals.get('infos', [])):
            if self._do_record_dict[env_idx]:
                self._read_info(env_idx, info)
        dones = list(enumerate(self.locals.get('dones', [])))
        dones.extend(list(enumerate(self.locals.get('done', []))))
        for env_idx, done in dones:
            if self._do_record_dict[env_idx]:
                self._read_done(env_idx, done)

        return True

    def _on_training_end(self) -> None:
        for env_idx in range(len(self._recorder_dict)):
            if self._recorder_dict[env_idx]:
                self._recorder_out_list.append({'steps': self._recorder_dict[env_idx],
                                                'episode': self._episode_counter})
        pass
