import sys
from pathlib import Path
from matplotlib import pyplot as plt

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

import time


import simplejson
from stable_baselines3.common.vec_env import SubprocVecEnv

from environments import helpers as h
from environments.factory.factory_dirt import DirtProperties, DirtFactory
from environments.factory.factory_dirt_item import DirtItemFactory
from environments.factory.factory_item import ItemProperties, ItemFactory
from environments.logging.monitor import MonitorCallback
from environments.utility_classes import MovementProperties
import pickle
from plotting.compare_runs import compare_seed_runs, compare_model_runs, compare_all_parameter_runs
import pandas as pd
import seaborn as sns

# Define a global studi save path
start_time = int(time.time())
study_root_path = Path(__file__).parent.parent / 'study_out' / f'{Path(__file__).stem}_{start_time}'

"""
In this studie, we want to explore the macro behaviour of multi agents which are trained on the same task, 
but never saw each other in training.
Those agents learned 


We start with training a single policy on a single task (dirt cleanup / item pickup).
Then multiple agent equipped with the same policy are deployed in the same environment.

There are further distinctions to be made:

1. No Observation - ['no_obs']:
- Agent do not see each other but their consequences of their combined actions
- Agents can collide

2. Observation in seperate slice - [['seperate_0'], ['seperate_1'], ['seperate_N']]:
- Agents see other entitys on a seperate slice
- This slice has been filled with $0 | 1 | \mathbb{N}(0, 1)$
-- Depending ob the fill value, agents will react diffently
   -> TODO: Test this! 

3. Observation in level slice - ['in_lvl_obs']:
- This tells the agent to treat other agents as obstacle. 
- However, the state space is altered since moving obstacles are not part the original agent observation. 
- We are out of distribution.

4. Obseration (similiar to camera read out) ['in_lvl_0.5', 'in_lvl_n']
- This tells the agent to treat other agents as obstacle, but "sees" them encoded as a different value. 
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
    train_steps = 5e5

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
        'seperate_0': dict(additional_env_kwargs=dict(additional_agent_placeholder=0)),
        #  Fill-value = 1
        'seperate_1': dict(additional_env_kwargs=dict(additional_agent_placeholder=1)),
        #  Fill-value = N(0, 1)
        'seperate_N': dict(additional_env_kwargs=dict(additional_agent_placeholder='N')),
        #  Further Adjustments are done post-training
        'in_lvl_obs': dict(post_training_kwargs=dict(other_agent_obs='in_lvl')),
        #  No further adjustment needed
        'no_obs': {}
    }

    # Train starts here ############################################################
    # Build Major Loop  parameters, parameter versions, Env Classes and models
    if True:
        for observation_mode in observation_modes.keys():
            for env_name in env_names:
                for model_cls in h.MODEL_MAP.values():
                    # Create an identifier, which is unique for every combination and easy to read in filesystem
                    identifier = f'{model_cls.__name__}_{start_time}'
                    # Train each combination per seed
                    combination_path = study_root_path / observation_mode / env_name / identifier
                    env_class, env_kwargs = env_map[env_name]
                    # Retrieve and set the observation mode specific env parameters
                    if observation_mode_kwargs := observation_modes.get(observation_mode, None):
                        if additional_env_kwargs := observation_mode_kwargs.get("additional_env_kwargs", None):
                            env_kwargs.update(additional_env_kwargs)
                    for seed in range(5):
                        env_kwargs.update(env_seed=seed)
                        # Output folder
                        seed_path = combination_path / f'{str(seed)}_{identifier}'
                        seed_path.mkdir(parents=True, exist_ok=True)

                        # Monitor Init
                        callbacks = [MonitorCallback(seed_path / 'monitor.pick')]

                        # Env Init & Model kwargs definition
                        if model_cls.__name__ in ["PPO", "A2C"]:
                            # env_factory = env_class(**env_kwargs)
                            env_factory = SubprocVecEnv([encapsule_env_factory(env_class, env_kwargs)
                                                         for _ in range(1)], start_method="spawn")
                            model_kwargs = policy_model_kwargs()

                        elif model_cls.__name__ in ["RegDQN", "DQN", "QRDQN"]:
                            with env_class(**env_kwargs) as env_factory:
                                model_kwargs = dqn_model_kwargs()

                        else:
                            raise NameError(f'The model "{model_cls.__name__}" has the wrong name.')

                        param_path = seed_path / f'env_params.json'
                        try:
                            env_factory.env_method('save_params', param_path)
                        except AttributeError:
                            env_factory.save_params(param_path)

                        # Model Init
                        model = model_cls("MlpPolicy", env_factory,
                                          verbose=1, seed=seed, device='cpu',
                                          **model_kwargs)

                        # Model train
                        model.learn(total_timesteps=int(train_steps), callback=callbacks)

                        # Model save
                        save_path = seed_path / f'model.zip'
                        model.save(save_path)

                        # Better be save then sorry: Clean up!
                        del env_factory, model
                        import gc
                        gc.collect()

                    # Compare performance runs, for each seed within a model
                    compare_seed_runs(combination_path)
                    # Better be save then sorry: Clean up!
                    del model_kwargs, env_kwargs
                    import gc
                    gc.collect()

                # Compare performance runs, for each model
                # FIXME: Check THIS!!!!
                compare_model_runs(study_root_path / observation_mode / env_name, f'{start_time}', 'step_reward')
                pass
            pass
        pass
    pass
    # Train ends here ############################################################
    exit()
    # Evaluation starts here #####################################################
    # First Iterate over every model and monitor "as trained"
    baseline_monitor_file = 'e_1_baseline_monitor.pick'
    if True:
        render = True
        for observation_mode in observation_modes:
            obs_mode_path = next(x for x in study_root_path.iterdir() if x.is_dir() and x.name == observation_mode)
            # For trained policy in study_root_path / identifier
            for env_path in [x for x in obs_mode_path.iterdir() if x.is_dir()]:
                for policy_path in [x for x in env_path.iterdir() if x. is_dir()]:
                    # Iteration
                    for seed_path in (y for y in policy_path.iterdir() if y.is_dir()):
                        # retrieve model class
                        for model_cls in (val for key, val in h.MODEL_MAP.items() if key in policy_path.name):
                            # Load both agents
                            model = model_cls.load(seed_path / 'model.zip')
                            # Load old env kwargs
                            with next(seed_path.glob('*.json')).open('r') as f:
                                env_kwargs = simplejson.load(f)
                            # Monitor Init
                            with MonitorCallback(filepath=seed_path / baseline_monitor_file) as monitor:
                                # Init Env
                                env_factory = env_map[env_path.name][0](**env_kwargs)
                                # Evaluation Loop for i in range(n Episodes)
                                for episode in range(100):
                                    obs = env_factory.reset()
                                    rew, done_bool = 0, False
                                    while not done_bool:
                                        action = model.predict(obs, deterministic=True)[0]
                                        env_state, step_r, done_bool, info_obj = env_factory.step(action)
                                        monitor.read_info(0, info_obj)
                                        rew += step_r
                                        if render:
                                            env_factory.render()
                                        if done_bool:
                                            monitor.read_done(0, done_bool)
                                            break
                                    print(f'Factory run {episode} done, reward is:\n    {rew}')
                                # Eval monitor outputs are automatically stored by the monitor object
                            del model, env_kwargs, env_factory
                            import gc

                            gc.collect()

    # Then iterate over every model and monitor "ood behavior" - "is it ood?"
    ood_monitor_file = 'e_1_monitor.pick'
    if True:
        for observation_mode in observation_modes:
            obs_mode_path = next(x for x in study_root_path.iterdir() if x.is_dir() and x.name == observation_mode)
            # For trained policy in study_root_path / identifier
            for env_path in [x for x in obs_mode_path.iterdir() if x.is_dir()]:
                for policy_path in [x for x in env_path.iterdir() if x. is_dir()]:
                    # FIXME: Pick random seed or iterate over available seeds
                    # First seed path version
                    # seed_path = next((y for y in policy_path.iterdir() if y.is_dir()))
                    # Iteration
                    for seed_path in (y for y in policy_path.iterdir() if y.is_dir()):
                        if (seed_path / f'e_1_monitor.pick').exists():
                            continue
                        # retrieve model class
                        for model_cls in (val for key, val in h.MODEL_MAP.items() if key in policy_path.name):
                            # Load both agents
                            models = [model_cls.load(seed_path / 'model.zip') for _ in range(2)]
                            # Load old env kwargs
                            with next(seed_path.glob('*.json')).open('r') as f:
                                env_kwargs = simplejson.load(f)
                                env_kwargs.update(
                                    n_agents=2, additional_agent_placeholder=None,
                                    **observation_modes[observation_mode].get('post_training_env_kwargs', {}))

                            # Monitor Init
                            with MonitorCallback(filepath=seed_path / ood_monitor_file) as monitor:
                                # Init Env
                                with env_map[env_path.name][0](**env_kwargs) as env_factory:
                                    # Evaluation Loop for i in range(n Episodes)
                                    for episode in range(50):
                                        obs = env_factory.reset()
                                        rew, done_bool = 0, False
                                        while not done_bool:
                                            actions = [model.predict(obs[i], deterministic=False)[0]
                                                       for i, model in enumerate(models)]
                                            env_state, step_r, done_bool, info_obj = env_factory.step(actions)
                                            monitor.read_info(0, info_obj)
                                            rew += step_r
                                            if done_bool:
                                                monitor.read_done(0, done_bool)
                                                break
                                        print(f'Factory run {episode} done, reward is:\n    {rew}')
                                    # Eval monitor outputs are automatically stored by the monitor object
                            del models, env_kwargs, env_factory
                            import gc

                            gc.collect()

    # Plotting
    if True:
        # TODO: Plotting
        df_list = list()
        for observation_folder in (x for x in study_root_path.iterdir() if x.is_dir()):
            for env_folder in (x for x in observation_folder.iterdir() if x.is_dir()):
                for model_folder in (x for x in env_folder.iterdir() if x.is_dir()):
                    # Gather per seed results in this list

                    for seed_folder in (x for x in model_folder.iterdir() if x.is_dir()):
                        for monitor_file in [baseline_monitor_file, ood_monitor_file]:

                            with (seed_folder / monitor_file).open('rb') as f:
                                monitor_df = pickle.load(f)

                            monitor_df = monitor_df.fillna(0)
                            monitor_df['seed'] = int(seed_folder.name.split('_')[0])
                            monitor_df['monitor'] = monitor_file.split('.')[0]
                            monitor_df['monitor'] = monitor_df['monitor'].astype(str)
                            monitor_df['env'] = env_folder.name

                            monitor_df['obs_mode'] = observation_folder.name
                            monitor_df['obs_mode'] = monitor_df['obs_mode'].astype(str)
                            monitor_df['model'] = model_folder.name.split('_')[0]


                            df_list.append(monitor_df)

        id_cols = ['monitor', 'env', 'obs_mode', 'model']

        df = pd.concat(df_list, ignore_index=True)
        df = df.fillna(0)

        for id_col in id_cols:
            df[id_col] = df[id_col].astype(str)

        df_grouped = df.groupby(id_cols + ['seed']
                                ).agg({key: 'sum' if "Agent" in key else 'mean' for key in df.columns
                                       if key not in (id_cols + ['seed'])})
        df_melted = df_grouped.reset_index().melt(id_vars=id_cols,
                                                  value_vars='step_reward', var_name="Measurement",
                                                  value_name="Score")

        c = sns.catplot(data=df_melted, x='obs_mode', hue='monitor', row='model', col='env', y='Score', sharey=False,
                        kind="box", height=4, aspect=.7, legend_out=True)
        c.set_xticklabels(rotation=65, horizontalalignment='right')
        plt.tight_layout(pad=2)
        plt.show()

    pass
