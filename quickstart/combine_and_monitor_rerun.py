import sys
from pathlib import Path

##############################################
# keep this for stand alone script execution #
##############################################
from environments.factory.base.base_factory import BaseFactory
from environments.logging.recorder import EnvRecorder

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

from environments import helpers as h
from environments.factory.additional.combined_factories import DestBatteryFactory
from environments.factory.additional.dest.factory_dest import DestFactory
from environments.factory.additional.dirt.factory_dirt import DirtFactory
from environments.factory.additional.item.factory_item import ItemFactory
from environments.helpers import ObservationTranslator, ActionTranslator
from environments.logging.envmonitor import EnvMonitor
from environments.utility_classes import ObservationProperties, AgentRenderOptions, MovementProperties


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
    factory_kwargs = dict(
        max_steps=400, parse_doors=True,
        level_name='rooms',
        doors_have_area=True, verbose=False,
        mv_prop=MovementProperties(allow_diagonal_movement=True,
                                   allow_square_movement=True,
                                   allow_no_op=False),
        obs_prop=ObservationProperties(
            frames_to_stack=3,
            cast_shadows=True,
            omit_agent_self=True,
            render_agents=AgentRenderOptions.LEVEL,
            additional_agent_placeholder=None,
        )
    )

    # Bundle both environments with global kwargs and parameters
    # Todo: find a better solution, like outo module loading
    env_map = {'DirtFactory': DirtFactory,
               'ItemFactory': ItemFactory,
               'DestFactory': DestFactory,
               'DestBatteryFactory': DestBatteryFactory
               }
    env_names = list(env_map.keys())

    # Put all your multi-seed agends in a single folder, we do not need specific names etc.
    available_models = dict()
    available_envs = dict()
    available_runs_kwargs = dict()
    available_runs_agents = dict()
    max_seed = 0
    # Define this folder
    combinations_path = Path('combinations')
    # Those are all differently trained combinations of mdoels, env and parameters
    for combination in (x for x in combinations_path.iterdir() if x.is_dir()):
        # These are all the models for this specific combination
        for model_run in (x for x in combination.iterdir() if x.is_dir()):
            model_name, env_name = model_run.name.split('_')[:2]
            if model_name not in available_models:
                available_models[model_name] = h.MODEL_MAP[model_name]
            if env_name not in available_envs:
                available_envs[env_name] = env_map[env_name]
            # Those are all available seeds
            for seed_run in (x for x in model_run.iterdir() if x.is_dir()):
                max_seed = max(int(seed_run.name.split('_')[0]), max_seed)
                # Read the env configuration from ROM
                with next(seed_run.glob('env_params.json')).open('r') as f:
                    env_kwargs = simplejson.load(f)
                available_runs_kwargs[seed_run.name] = env_kwargs
                # Read the trained model_path from ROM
                model_path = next(seed_run.glob('model.zip'))
                available_runs_agents[seed_run.name] = model_path

    # We start by combining all SAME MODEL CLASSES per available Seed, across ALL available ENVIRONMENTS.
    for model_name, model_cls in available_models.items():
        for seed in range(max_seed):
            combined_env_kwargs = dict()
            model_paths = list()
            comparable_runs = {key: val for key, val in available_runs_kwargs.items() if (
                    key.startswith(str(seed)) and model_name in key and key != 'key')
                               }
            for name, run_kwargs in comparable_runs.items():
                # Select trained agent as a candidate:
                model_paths.append(available_runs_agents[name])
                # Sort Env Kwars:
                for key, val in run_kwargs.items():
                    if key not in combined_env_kwargs:
                        combined_env_kwargs.update(dict(key=val))
                    else:
                        assert combined_env_kwargs[key] == val, "Check the combinations you try to make!"

            # Update and combine all kwargs to account for multiple agents etc.
            # We cannot capture all configuration cases!
            for key, val in factory_kwargs.items():
                if key not in combined_env_kwargs:
                    combined_env_kwargs[key] = val
                else:
                    assert combined_env_kwargs[key] == val
            combined_env_kwargs.update(n_agents=len(comparable_runs))

            with(type("CombinedEnv", tuple(available_envs.values()), {})(**combined_env_kwargs)) as combEnv:
                # EnvMonitor Init
                comb = f'comb_{model_name}_{seed}'
                comb_monitor_path = combinations_path / comb / f'{comb}_monitor.pick'
                comb_recorder_path = combinations_path / comb / f'{comb}_recorder.pick'
                comb_monitor_path.parent.mkdir(parents=True, exist_ok=True)

                monitoredCombEnv = EnvMonitor(combEnv, filepath=comb_monitor_path)
                # monitoredCombEnv = EnvRecorder(monitoredCombEnv, filepath=comb_monitor_path)

                # Evaluation starts here #####################################################
                # Load all models
                loaded_models = [available_models[model_name].load(model_path) for model_path in model_paths]
                obs_translators = ObservationTranslator(
                    monitoredCombEnv.named_observation_space,
                    *[agent.named_observation_space for agent in loaded_models],
                    placeholder_fill_value='n')
                act_translators = ActionTranslator(
                    monitoredCombEnv.named_action_space,
                    *(agent.named_action_space for agent in loaded_models)
                )

                for episode in range(50):
                    obs, _ = monitoredCombEnv.reset(), monitoredCombEnv.render()
                    rew, done_bool = 0, False
                    while not done_bool:
                        actions = []
                        for i, model in enumerate(loaded_models):
                            pred = model.predict(obs_translators.translate_observation(i, obs[i]))[0]
                            actions.append(act_translators.translate_action(i, pred))

                        obs, step_r, done_bool, info_obj = monitoredCombEnv.step(actions)

                        rew += step_r
                        monitoredCombEnv.render()
                        if done_bool:
                            break
                    print(f'Factory run {episode} done, reward is:\n    {rew}')
                # Eval monitor outputs are automatically stored by the monitor object
                # TODO: Plotting
                monitoredCombEnv.save_records(comb_monitor_path)
                monitoredCombEnv.save_run()
            pass
