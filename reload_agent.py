import warnings
from pathlib import Path

import yaml
from natsort import natsorted
from environments import helpers as h

from environments.factory.factory_dirt_item import DirtItemFactory
from environments.logging.recorder import RecorderCallback

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)


if __name__ == '__main__':

    model_name = 'PPO_1631187073'
    run_id = 0
    seed = 69
    out_path = Path(__file__).parent / 'study_out' / 'e_1_1631709932'/ 'no_obs' / 'itemdirt'/'A2C_1631709932' / '0_A2C_1631709932'
    model_path = out_path / model_name

    with (out_path / f'env_params.json').open('r') as f:
        env_kwargs = yaml.load(f, Loader=yaml.FullLoader)
        env_kwargs.update(verbose=False, env_seed=seed, record_episodes=True)

    this_model = out_path / 'model.zip'

    model_cls = next(val for key, val in h.MODEL_MAP.items() if key in model_name)
    model = model_cls.load(this_model)

    with RecorderCallback(filepath=Path() / 'recorder_out.json') as recorder:
        # Init Env
        with DirtItemFactory(**env_kwargs) as env:
            # Evaluation Loop for i in range(n Episodes)
            for episode in range(5):
                obs = env.reset()
                rew, done_bool = 0, False
                while not done_bool:
                    action = model.predict(obs, deterministic=False)[0]
                    env_state, step_r, done_bool, info_obj = env.step(action[0])
                    recorder.read_info(0, info_obj)
                    rew += step_r
                    if done_bool:
                        recorder.read_done(0, done_bool)
                        break
                print(f'Factory run {episode} done, reward is:\n    {rew}')
    print('all done')
