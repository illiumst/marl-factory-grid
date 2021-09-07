import json
from collections import defaultdict
from pathlib import Path
from typing import Union

import pandas as pd
from stable_baselines3.common.callbacks import BaseCallback

from environments.factory.base.base_factory import REC_TAC
from environments.helpers import IGNORED_DF_COLUMNS


# noinspection PyAttributeOutsideInit
class RecorderCallback(BaseCallback):

    def __init__(self, filepath: Union[str, Path], occupation_map: bool = False, trajectory_map: bool = False):
        super(RecorderCallback, self).__init__()
        self.trajectory_map = trajectory_map
        self.occupation_map = occupation_map
        self.filepath = Path(filepath)
        self._recorder_dict = defaultdict(dict)
        self._recorder_json_list = list()
        self.do_record: bool
        self.started = False
        self.closed = False

    def read_info(self, env_idx, info: dict):
        if info_dict := {key.replace(REC_TAC, ''): val for key, val in info.items() if key.startswith(f'{REC_TAC}')}:
            info_dict.update(episode=(self.num_timesteps + env_idx))
            self._recorder_dict[env_idx][len(self._recorder_dict[env_idx])] = info_dict
        else:
            pass
        return

    def read_done(self, env_idx, done):
        if done:
            self._recorder_json_list.append(json.dumps(self._recorder_dict[env_idx]))
            self._recorder_dict[env_idx] = dict()
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
                json_list = self._recorder_json_list
                json.dump(json_list, f, indent=4)

            if self.occupation_map:
                print('Recorder files were dumped to disk, now plotting the occupation map...')

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
