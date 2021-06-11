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

    model_name = 'PPO_1623052687'
    run_id = 0
    out_path = Path(__file__).parent / 'debug_out'
    model_path = out_path / model_name

    with (model_path / f'env_{model_name}.yaml').open('r') as f:
        env_kwargs = yaml.load(f, Loader=yaml.FullLoader)
    with SimpleFactory(level_name='rooms', **env_kwargs) as env:

        # Edit THIS:
        model_files = list(natsorted((model_path / f'{run_id}_{model_name}').rglob('model_*.zip')))
        this_model = model_files[0]

        model = PPO.load(this_model)
        evaluation_result = evaluate_policy(model, env, n_eval_episodes=100, deterministic=False, render=True)
        print(evaluation_result)


