import warnings
from pathlib import Path

import numpy as np
import yaml

from environments import helpers as h
from environments.helpers import Constants as c
from environments.factory.factory_dirt import DirtFactory
from environments.factory.combined_factories import DirtItemFactory
from environments.logging.recorder import RecorderCallback

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)


if __name__ == '__main__':

    model_name = 'A2C_ItsDirt'
    run_id = 0
    seed = 67
    n_agents = 1
    out_path = Path('study_out/e_1_ItsDirt/no_obs/dirt/A2C_ItsDirt/0_A2C_ItsDirt')
    model_path = out_path

    with (out_path / f'env_params.json').open('r') as f:
        env_kwargs = yaml.load(f, Loader=yaml.FullLoader)
        env_kwargs.update(additional_agent_placeholder=None, n_agents=n_agents)
        if gain_amount := env_kwargs.get('dirt_prop', {}).get('gain_amount', None):
            env_kwargs['dirt_prop']['max_spawn_amount'] = gain_amount
            del env_kwargs['dirt_prop']['gain_amount']

        env_kwargs.update(record_episodes=False)

    this_model = out_path / 'model.zip'

    model_cls = next(val for key, val in h.MODEL_MAP.items() if key in model_name)
    models = [model_cls.load(this_model) for _ in range(n_agents)]

    with RecorderCallback(filepath=Path() / 'recorder_out_DQN.json') as recorder:
        # Init Env
        with DirtFactory(**env_kwargs) as env:
            obs_shape = env.observation_space.shape
            # Evaluation Loop for i in range(n Episodes)
            for episode in range(5):
                env_state = env.reset()
                rew, done_bool = 0, False
                while not done_bool:
                    if n_agents > 1:
                        actions = [model.predict(
                            np.stack([env_state[i][j] for i in range(env_state.shape[0])]),
                            deterministic=True)[0] for j, model in enumerate(models)]
                    else:
                        actions = models[0].predict(env_state, deterministic=True)[0]
                    if any([agent.pos in [door.pos for door in env.unwrapped[c.DOORS]]
                            for agent in env.unwrapped[c.AGENT]]):
                        print('On Door')
                    env_state, step_r, done_bool, info_obj = env.step(actions)

                    recorder.read_info(0, info_obj)
                    rew += step_r
                    env.render()
                    if done_bool:
                        recorder.read_done(0, done_bool)
                        break
                print(f'Factory run {episode} done, reward is:\n    {rew}')
    print('all done')
