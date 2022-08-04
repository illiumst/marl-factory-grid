import sys
import time

from pathlib import Path
import simplejson

import stable_baselines3 as sb3

# This is needed, when you put this file in a subfolder.
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

from environments import helpers as h
from environments.factory.additional.dest.dest_util import DestModeOptions, DestProperties
from environments.factory.additional.btry.btry_util import BatteryProperties
from environments.logging.envmonitor import EnvMonitor
from environments.logging.recorder import EnvRecorder
from environments.factory.additional.combined_factories import DestBatteryFactory
from environments.utility_classes import MovementProperties, ObservationProperties, AgentRenderOptions

from plotting.compare_runs import compare_seed_runs

"""
Welcome to this quick start file. Here we will see how to:
    0. Setup I/O Paths
    1. Setup parameters for the environments (dirt-factory).
    2. Setup parameters for the agent training (SB3: PPO) and save metrics.
        Run the training.
    3. Save env and agent for later analysis.
    4. Load the agent from drive
    5. Rendering the env with a run of the trained agent.
    6. Plot metrics 
"""

if __name__ == '__main__':
    #########################################################
    # 0. Setup I/O Paths
    # Define some general parameters
    train_steps = 1e6
    n_seeds = 3
    model_class = sb3.PPO
    env_class = DestBatteryFactory

    env_params_json = 'env_params.json'

    # Define a global studi save path
    start_time = int(time.time())
    study_root_path = Path(__file__).parent.parent / 'study_out' / f'{Path(__file__).stem}_{start_time}'
    # Create an identifier, which is unique for every combination and easy to read in filesystem
    identifier = f'{model_class.__name__}_{env_class.__name__}_{start_time}'
    exp_path = study_root_path / identifier

    #########################################################
    # 1. Setup parameters for the environments (dirt-factory).


    # Define property object parameters.
    #  'ObservationProperties' are for specifying how the agent sees the env.
    obs_props = ObservationProperties(render_agents=AgentRenderOptions.NOT,  # Agents won`t be shown in the obs at all
                                      omit_agent_self=True,                  # This is default
                                      additional_agent_placeholder=None,     # We will not take care of future agents
                                      frames_to_stack=3,                     # To give the agent a notion of time
                                      pomdp_r=2                              # the agents view-radius
                                      )
    #  'MovementProperties' are for specifying how the agent is allowed to move in the env.
    move_props = MovementProperties(allow_diagonal_movement=True,   # Euclidean style (vertices)
                                    allow_square_movement=True,     # Manhattan (edges)
                                    allow_no_op=False)              # Pause movement (do nothing)

    #  'DirtProperties' control if and how dirt is spawned
    # TODO: Comments
    dest_props = DestProperties(
        n_dests              = 2,  # How many destinations are there
        dwell_time           = 0,  # How long does the agent need to "wait" on a destination
        spawn_frequency      = 0,
        spawn_in_other_zone  = True,  #
        spawn_mode           = DestModeOptions.DONE,
    )
    btry_props = BatteryProperties(
        initial_charge          = 0.9,  #
        charge_rate             = 0.4,  #
        charge_locations        = 3,  #
        per_action_costs        = 0.01,
        done_when_discharged    = True,
        multi_charge            = False,
    )

    #  These are the EnvKwargs for initializing the env class, holding all former parameter-classes
    # TODO: Comments
    factory_kwargs = dict(n_agents=1,
                          max_steps=400,
                          parse_doors=True,
                          level_name='rooms',
                          doors_have_area=True,  #
                          verbose=False,
                          mv_prop=move_props,    # See Above
                          obs_prop=obs_props,    # See Above
                          done_at_collision=True,
                          dest_prop=dest_props,
                          btry_prop=btry_props
                          )

    #########################################################
    # 2. Setup parameters for the agent training (SB3: PPO) and save metrics.
    agent_kwargs = dict()


    #########################################################
    # Run the Training
    for seed in range(n_seeds):
        # Make a copy if you want to alter things in the training loop; like the seed.
        env_kwargs = factory_kwargs.copy()
        env_kwargs.update(env_seed=seed)

        # Output folder
        seed_path = exp_path / f'{str(seed)}_{identifier}'
        seed_path.mkdir(parents=True, exist_ok=True)

        # Parameter Storage
        param_path = seed_path / env_params_json
        # Observation (measures) Storage
        monitor_path = seed_path / 'monitor.pick'
        recorder_path = seed_path / 'recorder.json'
        # Model save Path for the trained model
        model_save_path = seed_path / f'model.zip'

        # Env Init & Model kwargs definition
        with env_class(**env_kwargs) as env_factory:

            # EnvMonitor Init
            env_monitor_callback = EnvMonitor(env_factory)

            # EnvRecorder Init
            env_recorder_callback = EnvRecorder(env_factory, freq=int(train_steps / 400 / 10))

            # Model Init
            model = model_class("MlpPolicy", env_factory, verbose=1, seed=seed, device='cpu')

            # Model train
            model.learn(total_timesteps=int(train_steps), callback=[env_monitor_callback, env_recorder_callback])

            #########################################################
            # 3. Save env and agent for later analysis.
            #   Save the trained Model, the monitor (env measures) and the env parameters
            model.named_observation_space = env_factory.named_observation_space
            model.named_action_space = env_factory.named_action_space
            model.save(model_save_path)
            env_factory.save_params(param_path)
            env_monitor_callback.save_run(monitor_path)
            env_recorder_callback.save_records(recorder_path, save_occupation_map=False)

    # Compare performance runs, for each seed within a model
    try:
        compare_seed_runs(exp_path, use_tex=False)
    except ValueError:
        pass

    # Train ends here ############################################################

    # Evaluation starts here #####################################################
    # First Iterate over every model and monitor "as trained"
    print('Start Measurement Tracking')
    # For trained policy in study_root_path / identifier
    for policy_path in [x for x in exp_path.iterdir() if x.is_dir()]:

        # retrieve model class
        model_cls = next(val for key, val in h.MODEL_MAP.items() if key in policy_path.parent.name)
        # Load the agent agent
        model = model_cls.load(policy_path / 'model.zip', device='cpu')
        # Load old env kwargs
        with next(policy_path.glob(env_params_json)).open('r') as f:
            env_kwargs = simplejson.load(f)
            # Make the env stop ar collisions
            # (you only want to have a single collision per episode hence the statistics)
            env_kwargs.update(done_at_collision=True)

        # Init Env
        with env_class(**env_kwargs) as env_factory:
            monitored_env_factory = EnvMonitor(env_factory)

            # Evaluation Loop for i in range(n Episodes)
            for episode in range(100):
                # noinspection PyRedeclaration
                env_state = monitored_env_factory.reset()
                rew, done_bool = 0, False
                while not done_bool:
                    action = model.predict(env_state, deterministic=True)[0]
                    env_state, step_r, done_bool, info_obj = monitored_env_factory.step(action)
                    rew += step_r
                    if done_bool:
                        break
                print(f'Factory run {episode} done, reward is:\n    {rew}')
            monitored_env_factory.save_run(filepath=policy_path / 'eval_run_monitor.pick')
    print('Measurements Done')
