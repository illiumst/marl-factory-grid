import warnings
from pathlib import Path

import numpy as np
import yaml

from environments import helpers as h
from environments.factory.factory_dirt import DirtFactory
from environments.factory.factory_dirt_item import DirtItemFactory
from environments.logging.recorder import RecorderCallback

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)


if __name__ == '__main__':

    model_name = 'DQN_1631187073'
    run_id = 0
    seed = 69
    out_path = Path('debug_out/DQN_1635176929/0_DQN_1635176929')
    model_path = out_path

    with (out_path / f'env_params.json').open('r') as f:
        env_kwargs = yaml.load(f, Loader=yaml.FullLoader)
        env_kwargs.update(additional_agent_placeholder=None, n_agents=4)
        if gain_amount := env_kwargs.get('dirt_properties', {}).get('gain_amount', None):
            env_kwargs['dirt_properties']['max_spawn_amount'] = gain_amount
            del env_kwargs['dirt_properties']['gain_amount']

        env_kwargs.update(record_episodes=True)

    this_model = out_path / 'model.zip'

    model_cls = next(val for key, val in h.MODEL_MAP.items() if key in model_name)
    models = [model_cls.load(this_model) for _ in range(4)]

    with RecorderCallback(filepath=Path() / 'recorder_out_DQN.json') as recorder:
        # Init Env
        with DirtItemFactory(**env_kwargs) as env:
            obs_shape = env.observation_space.shape
            # Evaluation Loop for i in range(n Episodes)
            for episode in range(5):
                env_state = env.reset()
                rew, done_bool = 0, False
                while not done_bool:
                    actions = [model.predict(
                        np.stack([env_state[i][j] for i in range(env_state.shape[0])]),
                        deterministic=True)[0] for j, model in enumerate(models)]
                    env_state, step_r, done_bool, info_obj = env.step(actions)
                    recorder.read_info(0, info_obj)
                    rew += step_r
                    # env.render()
                    if done_bool:
                        recorder.read_done(0, done_bool)
                        break
                print(f'Factory run {episode} done, reward is:\n    {rew}')
    print('all done')
