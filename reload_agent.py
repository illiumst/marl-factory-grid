import pickle
import warnings
from pathlib import Path

from natsort import natsorted
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

from environments.factory.simple_factory import DirtProperties, SimpleFactory

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)


if __name__ == '__main__':

    out_path = Path(r'C:\Users\steff\projects\f_iks\debug_out\A2C_1622558379')
    with (out_path / f'env_{out_path.name}.pick').open('rb') as f:
        env_kwargs = pickle.load(f)
    env = SimpleFactory(allow_no_op=False, allow_diagonal_movement=False, allow_square_movement=True, **env_kwargs)

    # Edit THIS:
    model_path = out_path / '1_A2C_1622558379'

    model_files = list(natsorted(out_path.rglob('*.zip')))
    this_model = model_files[0]

    model = PPO.load(this_model)
    evaluation_result = evaluate_policy(model, env, n_eval_episodes=100, deterministic=False,
                                        render=True)
    print(evaluation_result)

    env.close()
