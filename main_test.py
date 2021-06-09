# foreign imports
import warnings

from pathlib import Path
import yaml
from natsort import natsorted

from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3 import PPO, DQN, A2C

# our imports
from environments.factory.simple_factory import SimpleFactory
from environments.logging.monitor import MonitorCallback
from algorithms.reg_dqn import RegDQN
from main import compare_runs, combine_runs

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
model_mapping = dict(A2C=A2C, PPO=PPO, DQN=DQN, RegDQN=RegDQN)


if __name__ == '__main__':

    # get n policies pi_1, ..., pi_n trained in single agent setting
    # rewards = []
    # repeat for x eval runs
    # total reward = rollout game for y steps with n policies in multi-agent setting
    # rewards += [total reward]
    # boxplot total rewards

    run_id = '1623078961'
    model_name = 'PPO'

    # -----------------------
    out_path = Path(__file__).parent / 'debug_out'

    # from sb3_contrib import QRDQN
    model_path = out_path / f'{model_name}_{run_id}'
    model_files = list(natsorted(model_path.rglob('model_*.zip')))
    this_model = model_files[0]
    render = True

    model = model_mapping[model_name].load(this_model)

    for seed in range(3):
        with (model_path / f'env_{model_path.name}.yaml').open('r') as f:
            env_kwargs = yaml.load(f, Loader=yaml.FullLoader)
        env_kwargs.update(n_agents=2)
        env = SimpleFactory(**env_kwargs)

        exp_out_path = model_path / 'exp'
        callbacks = CallbackList(
            [MonitorCallback(filepath=exp_out_path / f'future_exp_name', plotting=True)]
        )

        n_actions = env.action_space.n

        for epoch in range(100):
            observations = env.reset()
            if render:
                env.render()
            done_bool = False
            r = 0
            while not done_bool:
                actions = [model.predict(obs, deterministic=False)[0] for obs in observations]

                obs, r, done_bool, info_obj = env.step(actions)
                if render:
                    env.render()
                if done_bool:
                    break
            print(f'Factory run {epoch} done, reward is:\n    {r}')

    if out_path:
        combine_runs(out_path.parent)
