import warnings
from pathlib import Path
import time

from natsort import natsorted
from stable_baselines3 import PPO
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.evaluation import evaluate_policy

from environments.factory.simple_factory import DirtProperties, SimpleFactory
from environments.logging.monitor import MonitorCallback
from environments.logging.training import TraningMonitor

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)


if __name__ == '__main__':
    dirt_props = DirtProperties()
    env = SimpleFactory(n_agents=1, dirt_properties=dirt_props)

    out_path = Path('debug_out')
    model_files = list(natsorted(out_path.rglob('*.zip')))
    this_model = model_files[0]

    model = PPO.load(this_model)
    evaluation_result = evaluate_policy(model, env, n_eval_episodes=100, deterministic=False,
                                        render=True)
    print(evaluation_result)

    env.close()