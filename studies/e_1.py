import sys
from pathlib import Path
from matplotlib import pyplot as plt
import itertools as it

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

import simplejson
from stable_baselines3.common.vec_env import SubprocVecEnv

from environments import helpers as h
from environments.factory.factory_dirt import DirtFactory
from environments.factory.dirt_util import DirtProperties
from environments.factory.combined_factories import DirtItemFactory
from environments.factory.factory_item import ItemFactory
from environments.factory.additional.item.item_util import ItemProperties
from environments.logging.envmonitor import EnvMonitor
from environments.utility_classes import MovementProperties, ObservationProperties, AgentRenderOptions
import pickle
from plotting.compare_runs import compare_seed_runs, compare_model_runs
import pandas as pd
import seaborn as sns

import multiprocessing as mp

"""
In this studie, we want to explore the macro behaviour of multi agent which are trained on the same task, 
but never saw each other in training.
Those agent learned 


We start with training a single policy on a single task (dirt cleanup / item pickup).
Then multiple agent equipped with the same policy are deployed in the same environment.

There are further distinctions to be made:

1. No Observation - ['no_obs']:
- Agent do not see each other but their consequences of their combined actions
- Agents can collide

2. Observation in seperate slice - [['seperate_0'], ['seperate_1'], ['seperate_N']]:
- Agents see other entitys on a seperate slice
- This slice has been filled with $0 | 1 | \mathbb{N}(0, 1)$
-- Depending ob the fill value, agent will react diffently
   -> TODO: Test this! 

3. Observation in level slice - ['in_lvl_obs']:
- This tells the agent to treat other agent as obstacle. 
- However, the state space is altered since moving obstacles are not part the original agent observation. 
- We are out of distribution.

4. Obseration (similiar to camera read out) ['in_lvl_0.5', 'in_lvl_n']
- This tells the agent to treat other agent as obstacle, but "sees" them encoded as a different value. 
- However, the state space is altered since moving obstacles are not part the original agent observation. 
- We are out of distribution.
"""

n_agents = 4
ood_monitor_file = f'e_1_{n_agents}_agents'
baseline_monitor_file = 'e_1_baseline'


def policy_model_kwargs():
    return dict()  # gae_lambda=0.25, n_steps=16, max_grad_norm=0.25, use_rms_prop=True)


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


def load_model_run_baseline(seed_path, env_to_run):
    # retrieve model class
    model_cls = next(val for key, val in h.MODEL_MAP.items() if key in seed_path.parent.name)
    # Load both agent
    model = model_cls.load(seed_path / 'model.zip', device='cpu')
    # Load old env kwargs
    with next(seed_path.glob('*.json')).open('r') as f:
        env_kwargs = simplejson.load(f)
        env_kwargs.update(done_at_collision=True)
    # Init Env
    with env_to_run(**env_kwargs) as env_factory:
        monitored_env_factory = EnvMonitor(env_factory)

        # Evaluation Loop for i in range(n Episodes)
        for episode in range(100):
            env_state = monitored_env_factory.reset()
            rew, done_bool = 0, False
            while not done_bool:
                action = model.predict(env_state, deterministic=True)[0]
                env_state, step_r, done_bool, info_obj = monitored_env_factory.step(action)
                rew += step_r
                if done_bool:
                    break
            print(f'Factory run {episode} done, reward is:\n    {rew}')
        monitored_env_factory.save_run(filepath=seed_path / f'{baseline_monitor_file}.pick')


def load_model_run_study(seed_path, env_to_run, additional_kwargs_dict):
    global model_cls
    # retrieve model class
    model_cls = next(val for key, val in h.MODEL_MAP.items() if key in seed_path.parent.name)
    # Load both agent
    models = [model_cls.load(seed_path / 'model.zip', device='cpu') for _ in range(n_agents)]
    # Load old env kwargs
    with next(seed_path.glob('*.json')).open('r') as f:
        env_kwargs = simplejson.load(f)
        env_kwargs.update(
            n_agents=n_agents,
            done_at_collision=True,
            **additional_kwargs_dict.get('post_training_kwargs', {}))
    # Init Env
    with env_to_run(**env_kwargs) as env_factory:
        monitored_factory_env = EnvMonitor(env_factory)
        # Evaluation Loop for i in range(n Episodes)
        for episode in range(50):
            env_state = monitored_factory_env.reset()
            rew, done_bool = 0, False
            while not done_bool:
                try:
                    actions = [model.predict(env_state[model_idx], deterministic=True)[0]
                               for model_idx, model in enumerate(models)]
                except ValueError as e:
                    print(e)
                    print('Env_Kwargs are:\n')
                    print(env_kwargs)
                    print('Path is:\n')
                    print(seed_path)
                    exit()
                env_state, step_r, done_bool, info_obj = monitored_factory_env.step(actions)
                rew += step_r
                if done_bool:
                    break
            print(f'Factory run {episode} done, reward is:\n    {rew}')
        monitored_factory_env.save_run(filepath=seed_path / f'{ood_monitor_file}.pick')
    # Eval monitor outputs are automatically stored by the monitor object
    del models, env_kwargs, env_factory
    import gc
    gc.collect()


def start_mp_study_run(envs_map, policies_path):
    paths = list(y for y in policies_path.iterdir() if y.is_dir() and not (y / f'{ood_monitor_file}.pick').exists())
    if paths:
        with mp.get_context("spawn").Pool(mp.cpu_count()) as pool:
            print("Starting MP with: ", pool._processes, " Processes")
            _ = pool.starmap(load_model_run_study,
                             it.product(paths,
                                        (envs_map[policies_path.parent.name][0],),
                                        (observation_modes[policies_path.parent.parent.name],))
                             )


def start_mp_baseline_run(envs_map, policies_path):
    paths = list(y for y in policies_path.iterdir() if y.is_dir() and
                 not (y / f'{baseline_monitor_file}.pick').exists())
    if paths:
        with mp.get_context("spawn").Pool(mp.cpu_count()) as pool:
            print("Starting MP with: ", pool._processes, " Processes")
            _ = pool.starmap(load_model_run_baseline,
                             it.product(paths,
                                        (envs_map[policies_path.parent.name][0],))
                             )


if __name__ == '__main__':
    # What to do:
    train = True
    baseline_run = True
    ood_run = True
    plotting = True

    train_steps = 1e6
    n_seeds = 3
    frames_to_stack = 3

    # Define a global studi save path
    start_time = 'new_reward'  # int(time.time())
    study_root_path = Path(__file__).parent.parent / 'study_out' / f'{Path(__file__).stem}_{start_time}'

    # Define Global Env Parameters
    # Define properties object parameters
    obs_props = ObservationProperties(render_agents=AgentRenderOptions.NOT,
                                      omit_agent_self=True,
                                      additional_agent_placeholder=None,
                                      frames_to_stack=frames_to_stack,
                                      pomdp_r=2
                                      )
    move_props = MovementProperties(allow_diagonal_movement=True,
                                    allow_square_movement=True,
                                    allow_no_op=False)
    dirt_props = DirtProperties(initial_dirt_ratio=0.35, initial_dirt_spawn_r_var=0.1,
                                clean_amount=0.34,
                                max_spawn_amount=0.1, max_global_amount=20,
                                max_local_amount=1, spawn_frequency=0, max_spawn_ratio=0.05,
                                dirt_smear_amount=0.0)
    item_props = ItemProperties(n_items=10,
                                spawn_frequency=30, n_drop_off_locations=2,
                                max_agent_inventory_capacity=15)
    factory_kwargs = dict(n_agents=1, max_steps=400, parse_doors=True,
                          level_name='rooms', doors_have_area=True,
                          verbose=False,
                          mv_prop=move_props,
                          obs_prop=obs_props,
                          done_at_collision=True
                          )

    # Bundle both environments with global kwargs and parameters
    env_map = {}
    env_map.update({'dirt': (DirtFactory, dict(dirt_prop=dirt_props,
                                               **factory_kwargs.copy()))})
    if False:
        env_map.update({'item': (ItemFactory, dict(item_prop=item_props,
                                                   **factory_kwargs.copy()))})
        env_map.update({'itemdirt': (DirtItemFactory, dict(dirt_prop=dirt_props, item_prop=item_props,
                                                           **factory_kwargs.copy()))})
    env_names = list(env_map.keys())

    # Define parameter versions according with #1,2[1,0,N],3
    observation_modes = {}
    if False:
        observation_modes.update({
            'seperate_1': dict(
                post_training_kwargs=
                dict(obs_prop=ObservationProperties(
                    render_agents=AgentRenderOptions.COMBINED,
                    additional_agent_placeholder=None,
                    omit_agent_self=True,
                    frames_to_stack=frames_to_stack,
                    pomdp_r=2)
                ),
                additional_env_kwargs=
                dict(obs_prop=ObservationProperties(
                    render_agents=AgentRenderOptions.NOT,
                    additional_agent_placeholder=1,
                    omit_agent_self=True,
                    frames_to_stack=frames_to_stack,
                    pomdp_r=2)
                )
            )})
        observation_modes.update({
            'seperate_0': dict(
                post_training_kwargs=
                dict(obs_prop=ObservationProperties(
                    render_agents=AgentRenderOptions.COMBINED,
                    additional_agent_placeholder=None,
                    omit_agent_self=True,
                    frames_to_stack=frames_to_stack,
                    pomdp_r=2)
                ),
                additional_env_kwargs=
                dict(obs_prop=ObservationProperties(
                    render_agents=AgentRenderOptions.NOT,
                    additional_agent_placeholder=0,
                    omit_agent_self=True,
                    frames_to_stack=frames_to_stack,
                    pomdp_r=2)
                )
            )})
        observation_modes.update({
            'seperate_N': dict(
                post_training_kwargs=
                dict(obs_prop=ObservationProperties(
                    render_agents=AgentRenderOptions.COMBINED,
                    additional_agent_placeholder=None,
                    omit_agent_self=True,
                    frames_to_stack=frames_to_stack,
                    pomdp_r=2)
                ),
                additional_env_kwargs=
                dict(obs_prop=ObservationProperties(
                    render_agents=AgentRenderOptions.NOT,
                    additional_agent_placeholder='N',
                    omit_agent_self=True,
                    frames_to_stack=frames_to_stack,
                    pomdp_r=2)
                )
            )})
        observation_modes.update({
            'in_lvl_obs': dict(
                post_training_kwargs=
                dict(obs_prop=ObservationProperties(
                    render_agents=AgentRenderOptions.LEVEL,
                    omit_agent_self=True,
                    additional_agent_placeholder=None,
                    frames_to_stack=frames_to_stack,
                    pomdp_r=2)
                )
            )})
    observation_modes.update({
        #  No further adjustment needed
        'no_obs': dict(
            post_training_kwargs=
            dict(obs_prop=ObservationProperties(
                render_agents=AgentRenderOptions.NOT,
                additional_agent_placeholder=None,
                omit_agent_self=True,
                frames_to_stack=frames_to_stack,
                pomdp_r=2)
            )
        )
    })

    # Train starts here ############################################################
    # Build Major Loop  parameters, parameter versions, Env Classes and models
    if train:
        for obs_mode in observation_modes.keys():
            for env_name in env_names:
                for model_cls in [h.MODEL_MAP['A2C']]:
                    # Create an _identifier, which is unique for every combination and easy to read in filesystem
                    identifier = f'{model_cls.__name__}_{start_time}'
                    # Train each combination per seed
                    combination_path = study_root_path / obs_mode / env_name / identifier
                    env_class, env_kwargs = env_map[env_name]
                    env_kwargs = env_kwargs.copy()
                    # Retrieve and set the observation mode specific env parameters
                    additional_kwargs = observation_modes.get(obs_mode, {}).get("additional_env_kwargs", {})
                    env_kwargs.update(additional_kwargs)
                    for seed in range(n_seeds):
                        env_kwargs.update(env_seed=seed)
                        # Output folder
                        seed_path = combination_path / f'{str(seed)}_{identifier}'
                        if (seed_path / 'monitor.pick').exists():
                            continue
                        seed_path.mkdir(parents=True, exist_ok=True)

                        # Env Init & Model kwargs definition
                        if model_cls.__name__ in ["PPO", "A2C"]:
                            # env_factory = env_class(**env_kwargs)

                            env_factory = SubprocVecEnv([encapsule_env_factory(env_class, env_kwargs)
                                                         for _ in range(6)], start_method="spawn")
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

                        # EnvMonitor Init
                        callbacks = [EnvMonitor(env_factory)]

                        # Model Init
                        model = model_cls("MlpPolicy", env_factory,
                                          verbose=1, seed=seed, device='cpu',
                                          **model_kwargs)

                        # Model train
                        model.learn(total_timesteps=int(train_steps), callback=callbacks)

                        # Model save
                        save_path = seed_path / f'model.zip'
                        model.save(save_path)

                        # Monitor Save
                        callbacks[0].save_run(seed_path / 'monitor.pick')

                        # Better be save then sorry: Clean up!
                        del env_factory, model
                        import gc
                        gc.collect()

                    # Compare performance runs, for each seed within a model
                    try:
                        compare_seed_runs(combination_path, use_tex=False)
                    except ValueError:
                        pass
                    # Better be save then sorry: Clean up!
                    try:
                        del env_kwargs
                        del model_kwargs
                        import gc
                        gc.collect()
                    except NameError:
                        pass

                # Compare performance runs, for each model
                # FIXME: Check THIS!!!!
                try:
                    compare_model_runs(study_root_path / obs_mode / env_name, f'{start_time}', 'step_reward',
                                       use_tex=False)
                except ValueError:
                    pass
                pass
            pass
        pass
    pass
    # Train ends here ############################################################

    # Evaluation starts here #####################################################
    # First Iterate over every model and monitor "as trained"
    if baseline_run:
        print('Start Baseline Tracking')
        for obs_mode in observation_modes:
            obs_mode_path = next(x for x in study_root_path.iterdir() if x.is_dir() and x.name == obs_mode)
            # For trained policy in study_root_path / _identifier
            for env_path in [x for x in obs_mode_path.iterdir() if x.is_dir()]:
                for policy_path in [x for x in env_path.iterdir() if x. is_dir()]:
                    # Iteration
                    start_mp_baseline_run(env_map, policy_path)

                    # for policy_path in (y for y in policy_path.iterdir() if y.is_dir()):
                    #    load_model_run_baseline(policy_path)
        print('Baseline Tracking done')

    # Then iterate over every model and monitor "ood behavior" - "is it ood?"
    if ood_run:
        print('Start OOD Tracking')
        for obs_mode in observation_modes:
            obs_mode_path = next(x for x in study_root_path.iterdir() if x.is_dir() and x.name == obs_mode)
            # For trained policy in study_root_path / _identifier
            for env_path in [x for x in obs_mode_path.iterdir() if x.is_dir()]:
                for policy_path in [x for x in env_path.iterdir() if x. is_dir()]:
                    # FIXME: Pick random seed or iterate over available seeds
                    # First seed path version
                    # policy_path = next((y for y in policy_path.iterdir() if y.is_dir()))
                    # Iteration
                    start_mp_study_run(env_map, policy_path)
                    #for policy_path in (y for y in policy_path.iterdir() if y.is_dir()):
                    #    load_model_run_study(policy_path, env_map[env_path.name][0], observation_modes[obs_mode])
        print('OOD Tracking Done')

    # Plotting
    if plotting:
        # TODO: Plotting
        print('Start Plotting')
        df_list = list()
        for observation_folder in (x for x in study_root_path.iterdir() if x.is_dir()):

            for env_folder in (x for x in observation_folder.iterdir() if x.is_dir()):
                for model_folder in (x for x in env_folder.iterdir() if x.is_dir()):
                    # Gather per seed results in this list

                    for seed_folder in (x for x in model_folder.iterdir() if x.is_dir()):
                        for monitor_file in [f'{baseline_monitor_file}.pick', f'{ood_monitor_file}.pick']:

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
        for env_name in env_names:
            for id_col in id_cols:
                df[id_col] = df[id_col].astype(str)

                if True:
                    # df['fail_sum'] = df.loc[:, df.columns.str.contains("failed")].sum(1)
                    df['pick_up'] = df.loc[:, df.columns.str.contains("]_item_pickup")].sum(1)
                    df['drop_off'] = df.loc[:, df.columns.str.contains("]_item_dropoff")].sum(1)
                    df['failed_item_action'] = df.loc[:, df.columns.str.contains("]_failed_item_action")].sum(1)
                    df['failed_cleanup'] = df.loc[:, df.columns.str.contains("]_failed_dirt_cleanup")].sum(1)
                    df['coll_lvl'] = df.loc[:, df.columns.str.contains("]_vs_LEVEL")].sum(1)
                    df['coll_agent'] = df.loc[:, df.columns.str.contains("]_vs_Agent")].sum(1) / 2
                    # df['`collis`ions'] = df['coll_lvl'] + df['coll_agent']

                    value_vars = ['pick_up', 'drop_off', 'failed_item_action', 'failed_cleanup',
                                  'coll_lvl', 'coll_agent', 'dirt_cleaned']

                df_grouped = df.groupby(id_cols + ['seed']
                                        # 'sum' if "agent" in key else 'mean'
                                        ).agg({key: 'sum' for key in df.columns
                                               if key not in (id_cols + ['seed'])})
                df_melted = df_grouped.reset_index().melt(id_vars=id_cols,
                                                          value_vars=value_vars,  # 'step_reward',
                                                          var_name="Measurement",
                                                          value_name="Score")
                # df_melted["Measurements"] = df_melted["Measurement"] + " " + df_melted["monitor"]

                # Plotting
                fig, ax = plt.subplots(figsize=(11.7, 8.27))

                c = sns.catplot(data=df_melted[df_melted['env'] == env_name],
                                x='Measurement', hue='monitor', row='model', col='obs_mode', y='Score',
                                sharey=True, kind="box", height=4, aspect=.7, legend_out=False, legend=False,
                                showfliers=False)
                c.set_xticklabels(rotation=65, horizontalalignment='right')
                # c.fig.subplots_adjust(top=0.9)  # adjust the Figure in rp
                c.fig.suptitle(f"Cat plot for {env_name}")
                plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
                plt.tight_layout()
                plt.savefig(study_root_path / f'results_{n_agents}_agents_{env_name}.png')
        pass
