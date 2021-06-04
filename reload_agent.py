import warnings
from pathlib import Path

import yaml
from natsort import natsorted
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

from environments.factory.simple_factory import DirtProperties, SimpleFactory

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)


if __name__ == '__main__':

    model_name = 'A2C_1622650432'
    run_id = 0
    out_path = Path(__file__).parent / 'debug_out'
    model_path = out_path / model_name

    with Path(r'C:\Users\steff\projects\f_iks\debug_out\yaml.txt').open('r') as f:
        env_kwargs = yaml.load(f)
    env = SimpleFactory(**env_kwargs)

    # Edit THIS:
    model_files = list(natsorted((model_path / f'{run_id}_{model_name}').rglob('*.zip')))
    this_model = model_files[0]

    model = PPO.load(this_model)
    evaluation_result = evaluate_policy(model, env, n_eval_episodes=100, deterministic=False, render=True)
    print(evaluation_result)

    env.close()
