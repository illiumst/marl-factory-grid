import itertools
import sys
from pathlib import Path

##############################################
# keep this for stand alone script execution #
##############################################
import numpy as np

try:
    # noinspection PyUnboundLocalVariable
    if __package__ is None:
        DIR = Path(__file__).resolve().parent
        sys.path.insert(0, str(DIR.parent))
        __package__ = DIR.name
    else:
        DIR = None
except NameError:
    DIR = None
    pass
##############################################
##############################################
##############################################


import simplejson
from stable_baselines3.common.vec_env import SubprocVecEnv

from environments import helpers as h
from environments.factory.factory_dirt import DirtFactory
from environments.factory.dirt_util import DirtProperties
from environments.factory.combined_factories import DirtItemFactory
from environments.factory.factory_item import ItemFactory
from environments.factory.additional.item.item_util import ItemProperties
from environments.logging.envmonitor import MonitorCallback
from environments.utility_classes import MovementProperties
from plotting.compare_runs import compare_seed_runs, compare_model_runs, compare_all_parameter_runs

# Define a global studi save path
start_time = 1631709932  # int(time.time())
study_root_path = (Path('..') if not DIR else Path()) / 'study_out' / f'e_1_{start_time}'

"""
In this studie, we want to explore the macro behaviour of multi agent which are trained on different tasks 
in the same environment, but never saw each other in training.
Those agent learned to maximize the individual environmental goal in {dirt clean, item pickup, mixed}

We start with agent that have been trained on a single task (dirt cleanup / item pickup / mixed).
We then mix two agent with any policy.

There are further distinctions to be made:

1. No Observation - ['no_obs']:
- Agent do not see each other but their consequences of their combined actions
- Agents can collide

2. Observation in seperate slice - [['seperate_0'], ['seperate_1'], ['seperate_N']]:
- Agents see other entitys on a seperate slice
- This slice has been filled with $0 | 1 | \mathbb{N}(0, 1)$
-- Depending ob the fill value, agent will react differently
   -> TODO: Test this!

3. Observation in level slice - ['in_lvl_obs']:
- This tells the agent to treat other agent as obstacle. 
- However, the state space is altered since moving obstacles are not part the original agent observation. 
- We are out of distribution.
"""


def policy_model_kwargs():
    return dict(ent_coef=0.01)


def dqn_model_kwargs():
    return dict(buffer_size=50000,
                learning_starts=64,
                batch_size=64,
                target_update_interval=5000,
                exploration_fraction=0.25,
                exploration_final_eps=0.025
                )


def encapsule_env_factory(env_fctry, env_kwrgs):

    def _init():
        with env_fctry(**env_kwrgs) as init_env:
            return init_env

    return _init


if __name__ == '__main__':

    # Define Global Env Parameters
    # Define properties object parameters
    move_props = MovementProperties(allow_diagonal_movement=True,
                                    allow_square_movement=True,
                                    allow_no_op=False)
    dirt_props = DirtProperties(clean_amount=2, gain_amount=0.1, max_global_amount=20,
                                max_local_amount=1, spawn_frequency=15, max_spawn_ratio=0.05,
                                dirt_smear_amount=0.0, agent_can_interact=True)
    item_props = ItemProperties(n_items=10, agent_can_interact=True,
                                spawn_frequency=30, n_drop_off_locations=2,
                                max_agent_inventory_capacity=15)
    factory_kwargs = dict(n_agents=1,
                          pomdp_r=2, max_steps=400, parse_doors=False,
                          level_name='rooms', frames_to_stack=3,
                          omit_agent_in_obs=True, combin_agent_obs=True, record_episodes=False,
                          cast_shadows=True, doors_have_area=False, verbose=False,
                          movement_properties=move_props
                          )

    # Bundle both environments with global kwargs and parameters
    env_map = {'dirt': (DirtFactory, dict(dirt_properties=dirt_props, **factory_kwargs)),
               'item': (ItemFactory, dict(item_properties=item_props, **factory_kwargs)),
               'itemdirt': (DirtItemFactory, dict(dirt_properties=dirt_props, item_properties=item_props,
                                                  **factory_kwargs))}
    env_names = list(env_map.keys())

    # Define parameter versions according with #1,2[1,0,N],3
    observation_modes = {
        #  Fill-value = 0
        'seperate_0': dict(additional_env_kwargs=dict(additional_agent_placeholder=0),
                           post_training_env_kwargs=dict(omit_agent_in_obs=True,
                                                         combin_agent_obs=False)
                           ),
        #  Fill-value = 1
        'seperate_1': dict(additional_env_kwargs=dict(additional_agent_placeholder=1),
                           post_training_env_kwargs=dict(omit_agent_in_obs=True,
                                                         combin_agent_obs=False)
                           ),
        #  Fill-value = N(0, 1)
        'seperate_N': dict(additional_env_kwargs=dict(additional_agent_placeholder='N'),
                           post_training_env_kwargs=dict(omit_agent_in_obs=True,
                                                         combin_agent_obs=False)
                           ),
        #  Further ADjustments are done post-training
        'in_lvl_obs': dict(post_training_kwargs=dict(other_agent_obs='in_lvl'),
                           ),
        #  No further adjustment needed
        'no_obs': None
    }

    # Evaluation starts here #####################################################
    # Iterate Observation Modes
    for observation_mode in observation_modes:
        obs_mode_path = next(x for x in study_root_path.iterdir() if x.is_dir() and x.name == observation_mode)
        # For trained policy in study_root_path / _identifier
        for env_paths in itertools.combinations([x for x in obs_mode_path.iterdir() if x.is_dir()], 2):
            policy_path_zip = zip(*[[x for x in env_paths[i].iterdir() if x.is_dir()] for i in range(len(env_paths))])
            for policy_paths in policy_path_zip:
                # TODO: Pick random seed or iterate over available seeds
                # First seed path version
                # policy_path = next((y for y in policy_path.iterdir() if y.is_dir()))
                # Iteration
                seed_path_zip = zip(*[[y for y in policy_paths[i].iterdir() if y.is_dir()] for i in range(len(policy_paths))])
                for seed_paths in seed_path_zip:
                    # retrieve model class
                    for model_cls in (val for key, val in h.MODEL_MAP.items() if key in policy_paths[0].name):
                        # Load both agent
                        models = [model_cls.load(seed_paths[i] / 'model.zip') for i in range(len(seed_paths))]
                        # Load old env kwargs
                        with next(seed_paths[0].glob('*.json')).open('r') as f:
                            env_kwargs = simplejson.load(f)
                        # Update Kwargs to account for multiple agent etc.
                        env_kwargs.update(n_agents=len(models), additional_agent_placeholder=None,
                                          **observation_modes[observation_mode].get('post_training_env_kwargs', {}))
                        # EnvMonitor Init
                        comb = f'combination_{"_".join([env_paths[i].name for i in range(len(env_paths))])}'
                        comb_monitor_path = obs_mode_path / comb / 'e_1_mix_monitor.pick'
                        comb_monitor_path.parent.mkdir(parents=True, exist_ok=True)
                        with MonitorCallback(filepath=comb_monitor_path) as monitor:
                            # Init Env
                            env = env_map['itemdirt'][0](**env_kwargs)
                            # Evaluation Loop for i in range(n Episodes)
                            for episode in range(50):
                                obs = env.reset()
                                rew, done_bool = 0, False
                                while not done_bool:
                                    actions = []
                                    for i, model in enumerate(models):
                                        if ptk := observation_modes[observation_mode].get('post_training_kwargs', {}):
                                            if ptk.get('other_agent_obs', '') == 'in_lvl':
                                                a_obs = np.concatenate(
                                                    ((obs[i][0] + (obs[i][1] == 1).astype(np.float32))[None, ...],
                                                     obs[i][2:])
                                                )
                                            else:
                                                a_obs = obs[i]
                                        else:
                                            a_obs = obs[i]
                                    actions.append(model.predict(obs[i], deterministic=False)[0])
                                    env_state, step_r, done_bool, info_obj = env.step(actions)
                                    monitor._read_info(0, info_obj)
                                    rew += step_r
                                    if done_bool:
                                        monitor._read_done(0, done_bool)
                                        break
                                print(f'Factory run {episode} done, reward is:\n    {rew}')
                            # Eval monitor outputs are automatically stored by the monitor object
                            # TODO: Plotting
        pass
