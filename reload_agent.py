import warnings
from pathlib import Path

import yaml

from marl_factory_grid.environment.factory import Factory
from marl_factory_grid.utils.logging.envmonitor import EnvMonitor
from marl_factory_grid.utils.logging.recorder import EnvRecorder

from marl_factory_grid.modules.doors import constants as d

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)


if __name__ == '__main__':

    determin = False
    render = True
    record = False
    verbose = True
    seed = 13
    n_agents = 1
    # out_path = Path('study_out/e_1_new_reward/no_obs/dirt/A2C_new_reward/0_A2C_new_reward')
    out_path = Path('quickstart/combinations/single_agent_train_dirt_env_1659374984/PPO_DirtFactory_1659374984/0_PPO_DirtFactory_1659374984/')
    model_path = out_path

    with (out_path / f'env_params.json').open('r') as f:
        env_kwargs = yaml.load(f, Loader=yaml.FullLoader)
        env_kwargs.update(n_agents=n_agents, done_at_collision=False, verbose=verbose)

    this_model = out_path / 'model.zip'

    model_cls = None  # next(val for key, val in h.MODEL_MAP.items() if key in out_path.parent.name)
    models = [model_cls.load(this_model)]
    try:
        # Legacy Cleanups
        del env_kwargs['dirt_prop']['agent_can_interact']
        env_kwargs['verbose'] = True
    except KeyError:
        pass

    # Init Env
    with Factory(**env_kwargs) as env:
        env = EnvMonitor(env)
        env = EnvRecorder(env) if record else env
        obs_shape = env.observation_space.shape
        # Evaluation Loop for i in range(n Episodes)
        for episode in range(500):
            env_state = env.reset()
            rew, done_bool = 0, False
            while not done_bool:
                if n_agents > 1:
                    actions = [model.predict(env_state[model_idx], deterministic=determin)[0]
                               for model_idx, model in enumerate(models)]
                else:
                    actions = models[0].predict(env_state, deterministic=determin)[0]
                env_state, step_r, done_bool, info_obj = env.step(actions)

                rew += step_r
                if render:
                    env.render()
                try:
                    door = next(x for x in env.unwrapped.unwrapped[d.DOORS] if x.is_open)
                    print('openDoor found')
                except StopIteration:
                    pass

                if done_bool:
                    break
            print(f'Factory run {episode} done, steps taken {env.unwrapped.unwrapped._steps}, reward is:\n    {rew}')
        env.save_run(out_path / 'reload_monitor.pick',
                     auto_plotting_keys=['step_reward', 'cleanup_valid', 'cleanup_fail'])
        if record:
            env.save_records(out_path / 'reload_recorder.pick', save_occupation_map=True)
    print('all done')
