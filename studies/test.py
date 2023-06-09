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
from environments.logging.recorder import EnvRecorder
from environments import helpers as h
from environments.factory.factory_dirt import DirtFactory
from environments.factory.dirt_util import DirtProperties
from environments.factory.factory_item import ItemFactory
from environments.factory.additional.item.item_util import ItemProperties
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
        recorded_env_factory.save_run(filepath=policy_path / f'monitor.pick')
        recorded_env_factory.save_records(filepath=policy_path / f'recorder.json')



if __name__ == '__main__':
    # What to do:
    train = True
    individual_run = True
    combined_run = False
    multi_env = False

    train_steps = 2e6
    frames_to_stack = 3

    # Define a global studi save path
    study_root_path = Path(__file__).parent.parent / 'study_out' / f'{Path(__file__).stem}'

    def policy_model_kwargs():
        return dict()

    # Define Global Env Parameters
    # Define properties object parameters
    obs_props = ObservationProperties(render_agents=AgentRenderOptions.NOT,
                                      additional_agent_placeholder=None,
                                      omit_agent_self=True,
                                      frames_to_stack=frames_to_stack,
                                      pomdp_r=2, cast_shadows=True)
    move_props = MovementProperties(allow_diagonal_movement=True,
                                    allow_square_movement=True,
                                    allow_no_op=False)
    dirt_props = DirtProperties(initial_dirt_ratio=0.35, initial_dirt_spawn_r_var=0.1,
                                clean_amount=0.34,
                                max_spawn_amount=0.1, max_global_amount=20,
                                max_local_amount=1, spawn_frequency=0, max_spawn_ratio=0.05,
                                dirt_smear_amount=0.0, agent_can_interact=True)
    item_props = ItemProperties(n_items=10, spawn_frequency=30, n_drop_off_locations=2,
                                max_agent_inventory_capacity=15)

    factory_kwargs = dict(n_agents=1, max_steps=500, parse_doors=True,
                          level_name='rooms', doors_have_area=True,
                          verbose=False,
                          mv_prop=move_props,
                          obs_prop=obs_props,
                          done_at_collision=False
                          )

    # Bundle both environments with global kwargs and parameters
    env_map = {}
    env_map.update({'dirt': (DirtFactory, dict(dirt_prop=dirt_props,
                                               **factory_kwargs.copy()))})
    env_map.update({'item': (ItemFactory, dict(item_prop=item_props,
                                               **factory_kwargs.copy()))})
    # env_map.update({'dest': (DestFactory, dict(dest_prop=dest_props,
    #                                           **factory_kwargs.copy()))})

    env_names = list(env_map.keys())

    # Train starts here ############################################################
    # Build Major Loop  parameters, parameter versions, Env Classes and models
    if train:
        for env_key in (env_key for env_key in env_map if 'combined' != env_key):
            model_cls = h.MODEL_MAP['A2C']
            combination_path = study_root_path / env_key
            env_class, env_kwargs = env_map[env_key]

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
            callbacks = [EnvMonitor(env_factory)]

            # Model Init
            model = model_cls("MlpPolicy", env_factory, **policy_model_kwargs(),
                              verbose=1, seed=69, device='cpu')

            # Model train
            model.learn(total_timesteps=int(train_steps), callback=callbacks)

            # Model save

            save_path = combination_path / f'model.zip'
            model.save(save_path)

            # Monitor Save
            callbacks[0].save_run(combination_path / 'monitor.pick')

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

