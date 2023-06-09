import itertools
import sys
from pathlib import Path

from stable_baselines3.common.vec_env import SubprocVecEnv

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
from environments.helpers import ActionTranslator, ObservationTranslator
from environments.logging.recorder import EnvRecorder
from environments import helpers as h
from environments.factory.factory_dirt import DirtFactory
from environments.factory.dirt_util import DirtProperties
from environments.factory.factory_item import ItemFactory
from environments.factory.additional.item.item_util import ItemProperties
from environments.factory.factory_dest import DestFactory
from environments.factory.additional.dest.dest_util import DestModeOptions, DestProperties
from environments.factory.combined_factories import DirtDestItemFactory
from environments.logging.envmonitor import EnvMonitor
from environments.utility_classes import MovementProperties, ObservationProperties, AgentRenderOptions

"""
In this studie, we want to export trained Agents for debugging purposes.
"""


def encapsule_env_factory(env_fctry, env_kwrgs):

    def _init():
        with env_fctry(**env_kwrgs) as init_env:
            return init_env

    return _init


def load_model_run_baseline(policy_path, env_to_run):
    # retrieve model class
    model_cls = h.MODEL_MAP['A2C']
    # Load both agent
    model = model_cls.load(policy_path / 'model.zip', device='cpu')
    # Load old env kwargs
    with next(policy_path.glob('*params.json')).open('r') as f:
        env_kwargs = simplejson.load(f)
        env_kwargs.update(done_at_collision=True)
    # Init Env
    with env_to_run(**env_kwargs) as env_factory:
        monitored_env_factory = EnvMonitor(env_factory)
        recorded_env_factory = EnvRecorder(monitored_env_factory)

        # Evaluation Loop for i in range(n Episodes)
        for episode in range(5):
            env_state = recorded_env_factory.reset()
            rew, done_bool = 0, False
            while not done_bool:
                action = model.predict(env_state, deterministic=True)[0]
                env_state, step_r, done_bool, info_obj = recorded_env_factory.step(action)
                rew += step_r
                if done_bool:
                    break
            print(f'Factory run {episode} done, reward is:\n    {rew}')
        recorded_env_factory.save_run(filepath=policy_path / f'baseline_monitor.pick')
        recorded_env_factory.save_records(filepath=policy_path / f'baseline_recorder.json')


def load_model_run_combined(root_path, env_to_run, env_kwargs):
    # retrieve model class
    model_cls = h.MODEL_MAP['A2C']
    # Load both agent
    models = [model_cls.load(model_zip, device='cpu') for model_zip in root_path.rglob('model.zip')]
    # Load old env kwargs
    env_kwargs = env_kwargs.copy()
    env_kwargs.update(
        n_agents=len(models),
        done_at_collision=False)

    # Init Env
    with env_to_run(**env_kwargs) as env_factory:

        action_translator = ActionTranslator(env_factory.named_action_space,
                                             *[x.named_action_space for x in models])
        observation_translator = ObservationTranslator(env_factory.observation_space.shape[-2:],
                                                       env_factory.named_observation_space,
                                                       *[x.named_observation_space for x in models])

        env = EnvMonitor(env_factory)
        # Evaluation Loop for i in range(n Episodes)
        for episode in range(5):
            env_state = env.reset()
            rew, done_bool = 0, False
            while not done_bool:
                translated_observations = observation_translator(env_state)
                actions = [model.predict(translated_observations[model_idx], deterministic=True)[0]
                           for model_idx, model in enumerate(models)]
                translated_actions = action_translator(actions)
                env_state, step_r, done_bool, info_obj = env.step(translated_actions)
                rew += step_r
                if done_bool:
                    break
            print(f'Factory run {episode} done, reward is:\n    {rew}')
        env.save_run(filepath=root_path / f'monitor_combined.pick')
        # env.save_records(filepath=root_path / f'recorder_combined.json')


if __name__ == '__main__':
    # What to do:
    train = True
    individual_run = False
    combined_run = False
    multi_env = False

    train_steps = 1e6
    frames_to_stack = 3

    # Define a global studi save path
    paremters_of_interest = dict(
        show_global_position_info=[True, False],
        pomdp_r=[3],
        cast_shadows=[True, False],
        allow_diagonal_movement=[True],
        parse_doors=[True, False],
        doors_have_area=[True, False],
        done_at_collision=[True, False]
    )
    keys, vals = zip(*paremters_of_interest.items())

    # Then we find all permutations for those values
    p = list(itertools.product(*vals))

    # Finally we can create out list of dicts
    result = [{keys[index]: entry[index] for index in range(len(entry))} for entry in p]

    for u in result:
        file_name = '_'.join('_'.join([str(y)[0] for y in x]) for x in u.items())
        study_root_path = Path(__file__).parent.parent / 'study_out' / file_name

        # Model Kwargs
        policy_model_kwargs = dict(ent_coef=0.01)

        # Define Global Env Parameters
        # Define properties object parameters
        obs_props = ObservationProperties(render_agents=AgentRenderOptions.NOT,
                                          additional_agent_placeholder=None,
                                          omit_agent_self=True,
                                          frames_to_stack=frames_to_stack,
                                          pomdp_r=u['pomdp_r'], cast_shadows=u['cast_shadows'],
                                          show_global_position_info=u['show_global_position_info'])
        move_props = MovementProperties(allow_diagonal_movement=u['allow_diagonal_movement'],
                                        allow_square_movement=True,
                                        allow_no_op=False)
        dirt_props = DirtProperties(initial_dirt_ratio=0.35, initial_dirt_spawn_r_var=0.1,
                                    clean_amount=0.34,
                                    max_spawn_amount=0.1, max_global_amount=20,
                                    max_local_amount=1, spawn_frequency=0, max_spawn_ratio=0.05,
                                    dirt_smear_amount=0.0)
        item_props = ItemProperties(n_items=10, spawn_frequency=30, n_drop_off_locations=2,
                                    max_agent_inventory_capacity=15)
        dest_props = DestProperties(n_dests=4, spawn_mode=DestModeOptions.GROUPED, spawn_frequency=1)
        factory_kwargs = dict(n_agents=1, max_steps=500, parse_doors=u['parse_doors'],
                              level_name='rooms', doors_have_area=u['doors_have_area'],
                              verbose=False,
                              mv_prop=move_props,
                              obs_prop=obs_props,
                              done_at_collision=u['done_at_collision']
                              )

        # Bundle both environments with global kwargs and parameters
        env_map = {}
        env_map.update({'dirt': (DirtFactory, dict(dirt_prop=dirt_props,
                                                   **factory_kwargs.copy()),
                                 ['cleanup_valid', 'cleanup_fail'])})
        # env_map.update({'item': (ItemFactory, dict(item_prop=item_props,
        #                                            **factory_kwargs.copy()),
        #                          ['DROPOFF_FAIL', 'ITEMACTION_FAIL', 'DROPOFF_VALID', 'ITEMACTION_VALID'])})
        # env_map.update({'dest': (DestFactory, dict(dest_prop=dest_props,
        #                                           **factory_kwargs.copy()))})
        env_map.update({'combined': (DirtDestItemFactory, dict(dest_prop=dest_props,
                                                               item_prop=item_props,
                                                               dirt_prop=dirt_props,
                                                               **factory_kwargs.copy()))})
        env_names = list(env_map.keys())

        # Train starts here ############################################################
        # Build Major Loop  parameters, parameter versions, Env Classes and models
        if train:
            for env_key in (env_key for env_key in env_map if 'combined' != env_key):
                model_cls = h.MODEL_MAP['PPO']
                combination_path = study_root_path / env_key
                env_class, env_kwargs, env_plot_keys = env_map[env_key]

                # Output folder
                if (combination_path / 'monitor.pick').exists():
                    continue
                combination_path.mkdir(parents=True, exist_ok=True)

                if not multi_env:
                    env_factory = encapsule_env_factory(env_class, env_kwargs)()
                else:
                    env_factory = SubprocVecEnv([encapsule_env_factory(env_class, env_kwargs)
                                                 for _ in range(6)], start_method="spawn")

                param_path = combination_path / f'env_params.json'
                try:
                    env_factory.env_method('save_params', param_path)
                except AttributeError:
                    env_factory.save_params(param_path)

                # EnvMonitor Init
                env_monitor = EnvMonitor(env_factory)
                callbacks = [env_monitor]

                # Model Init
                model = model_cls("MlpPolicy", env_factory, **policy_model_kwargs,
                                  verbose=1, seed=69, device='cpu')

                # Model train
                model.learn(total_timesteps=int(train_steps), callback=callbacks)

                # Model save
                try:
                    model.named_action_space = env_factory.unwrapped.named_action_space
                    model.named_observation_space = env_factory.unwrapped.named_observation_space
                except AttributeError:
                    model.named_action_space = env_factory.get_attr("named_action_space")[0]
                    model.named_observation_space = env_factory.get_attr("named_observation_space")[0]
                save_path = combination_path / f'model.zip'
                model.save(save_path)

                # Monitor Save
                env_monitor.save_run(combination_path / 'monitor.pick',
                                      auto_plotting_keys=['step_reward', 'collision'] + env_plot_keys)

                # Better be save then sorry: Clean up!
                del env_factory, model
                import gc
                gc.collect()

    # Train ends here ############################################################

    # Evaluation starts here #####################################################
    # First Iterate over every model and monitor "as trained"
    if individual_run:
        print('Start Individual Recording')
        for env_key in (env_key for env_key in env_map if 'combined' != env_key):
            # For trained policy in study_root_path / _identifier
            policy_path = study_root_path / env_key
            load_model_run_baseline(policy_path, env_map[policy_path.name][0])

            # for policy_path in (y for y in policy_path.iterdir() if y.is_dir()):
            #    load_model_run_baseline(policy_path)
        print('Done Individual Recording')

    # Then iterate over every model and monitor "ood behavior" - "is it ood?"
    if combined_run:
        print('Start combined run')
        for env_key in (env_key for env_key in env_map if 'combined' == env_key):
            # For trained policy in study_root_path / _identifier
            factory, kwargs = env_map[env_key]
            load_model_run_combined(study_root_path, factory, kwargs)
        print('OOD Tracking Done')
