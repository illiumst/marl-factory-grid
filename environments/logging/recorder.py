import json
from pathlib import Path
from typing import Union

import pandas as pd
from stable_baselines3.common.callbacks import BaseCallback

from environments.factory.base.base_factory import REC_TAC
from environments.helpers import IGNORED_DF_COLUMNS


class RecorderCallback(BaseCallback):

    def __init__(self, filepath: Union[str, Path], occupation_map: bool = False, trajectory_map: bool = False):
        super(RecorderCallback, self).__init__()
        self.trajectory_map = trajectory_map
        self.occupation_map = occupation_map
        self.filepath = Path(filepath)
        self._recorder_dict = dict()
        self._recorder_df = pd.DataFrame()
        self.do_record: bool
        self.started = False
        self.closed = False

    def _on_step(self) -> bool:
        if self.do_record and self.started:
            for _, info in enumerate(self.locals.get('infos', [])):
                self._recorder_dict[self.num_timesteps] = {key: val for key, val in info.items()
                                                           if not key.startswith(f'{REC_TAC}_')}

            for env_idx, done in list(enumerate(self.locals.get('dones', []))) + \
                                 list(enumerate(self.locals.get('done', []))):
                if done:
                    env_monitor_df = pd.DataFrame.from_dict(self._recorder_dict, orient='index')
                    self._recorder_dict = dict()
                    columns = [col for col in env_monitor_df.columns if col not in IGNORED_DF_COLUMNS]
                    env_monitor_df = env_monitor_df.aggregate(
                        {col: 'mean' if col.endswith('ount') else 'sum' for col in columns}
                    )
                    env_monitor_df['episode'] = len(self._recorder_df)
                    self._recorder_df = self._recorder_df.append([env_monitor_df])
                else:
                    pass
        else:
            pass
        return True

    def __enter__(self):
        self._on_training_start()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._on_training_end()

    def _on_training_start(self) -> None:
        if self.started:
            pass
        else:
            if hasattr(self.training_env, 'record_episodes'):
                if self.training_env.record_episodes:
                    self.do_record = True
                    self.filepath.parent.mkdir(exist_ok=True, parents=True)
                    self.started = True
                else:
                    self.do_record = False
            else:
                self.do_record = False
        pass

    def _on_training_end(self) -> None:
        if self.closed:
            pass
        else:
            if self.do_record and self.started:
                # self.out_file.unlink(missing_ok=True)
                with self.filepath.open('w') as f:
                    json_df = self._recorder_df.to_json(orient="table")
                    parsed = json.loads(json_df)
                    json.dump(parsed, f, indent=4)

                if self.occupation_map:
                    print('Recorder files were dumped to disk, now plotting the occupation map...')

                if self.trajectory_map:
                    print('Recorder files were dumped to disk, now plotting the occupation map...')

                self.closed = True
                self.started = False
            else:
                pass
