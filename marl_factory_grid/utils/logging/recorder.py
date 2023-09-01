from os import PathLike
from pathlib import Path
from typing import Union, List

import yaml
from gymnasium import Wrapper

import numpy as np
import pandas as pd


class EnvRecorder(Wrapper):

    def __init__(self, env, filepath: Union[str, PathLike] = None,
                 episodes: Union[List[int], None] = None):
        super(EnvRecorder, self).__init__(env)
        self.filepath = filepath
        self.episodes = episodes
        self._curr_episode = 0
        self._curr_ep_recorder = list()
        self._recorder_out_list = list()

    def reset(self):
        self._curr_ep_recorder = list()
        self._recorder_out_list = list()
        self._curr_episode += 1
        return self.env.reset()

    def step(self, actions):
        obs_type, obs, reward, done, info = self.env.step(actions)
        if not self.episodes or self._curr_episode in self.episodes:
            summary: dict = self.env.summarize_state()
            # summary.update(done=done)
            # summary.update({'episode': self._curr_episode})
            # TODO Protobuff Adjustments                 ######
            # summary.update(info)
            self._curr_ep_recorder.append(summary)
            if done:
                self._recorder_out_list.append({'steps': self._curr_ep_recorder,
                                                'episode_nr': self._curr_episode})
                self._curr_ep_recorder = list()
        return obs_type, obs, reward, done, info

    def _finalize(self):
        if self._curr_ep_recorder:
            self._recorder_out_list.append({'steps': self._curr_ep_recorder.copy(),
                                            'episode_nr': len(self._recorder_out_list)})

    def save_records(self, filepath: Union[Path, str, None] = None,
                     only_deltas=False,
                     save_occupation_map=False,
                     save_trajectory_map=False,
                     ):
        self._finalize()
        filepath = Path(filepath or self.filepath)
        filepath.parent.mkdir(exist_ok=True, parents=True)
        # cls.out_file.unlink(missing_ok=True)
        with filepath.open('wb') as f:
            if only_deltas:
                from deepdiff import DeepDiff
                diff_dict = [DeepDiff(t1, t2, ignore_order=True)
                             for t1, t2 in zip(self._recorder_out_list, self._recorder_out_list[1:])
                             ]
                out_dict = {'episodes': diff_dict}

            else:
                # TODO Protobuff Adjustments Revert
                dest_prop = dict(
                    n_dests=0,
                    dwell_time=0,
                    spawn_frequency=0,
                    spawn_in_other_zone=False,
                    spawn_mode=''
                )
                rewards_dest = dict(
                    WAIT_VALID=0.00,
                    WAIT_FAIL=0.00,
                    DEST_REACHED=0.00,
                )
                mv_prop = dict(
                    allow_square_movement=False,
                    allow_diagonal_movement=False,
                    allow_no_op=False,
                )
                obs_prop = dict(
                    render_agents='',
                    omit_agent_self=False,
                    additional_agent_placeholder=0,
                    cast_shadows=False,
                    frames_to_stack=0,
                    pomdp_r=self.env.params['General']['pomdp_r'],
                    indicate_door_area=False,
                    show_global_position_info=False,

                )
                rewards_base = dict(
                    MOVEMENTS_VALID=0.00,
                    MOVEMENTS_FAIL=0.00,
                    NOOP=0.00,
                    USE_DOOR_VALID=0.00,
                    USE_DOOR_FAIL=0.00,
                    COLLISION=0.00,

                )

                out_dict = {'episodes': self._recorder_out_list}
            out_dict.update(
                {'n_episodes': self._curr_episode,
                 'metadata':dict(
                     level_name=self.env.params['General']['level_name'],
                     verbose=False,
                     n_agents=len(self.env.params['Agents']),
                     max_steps=100,
                     done_at_collision=False,
                     parse_doors=True,
                     doors_have_area=False,
                     individual_rewards=True,
                     class_name='Where does this end up?',
                     env_seed=69,

                     dest_prop=dest_prop,
                     rewards_dest=rewards_dest,
                     mv_prop=mv_prop,
                     obs_prop=obs_prop,
                     rewards_base=rewards_base,
                 ),
                 # 'env_params': self.env.params,
                 'header': self.env.summarize_header()
                 })
            try:
                from marl_factory_grid.utils.proto import fiksProto_pb2
                from google.protobuf import json_format

                bulk = fiksProto_pb2.Bulk()
                json_format.ParseDict(out_dict, bulk)
                f.write(bulk.SerializeToString())
                # yaml.dump(out_dict, f, indent=4)
            except TypeError:
                print('Shit')
        print('done')

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
