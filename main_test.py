# foreign imports
import warnings

from pathlib import Path
import yaml
from gym.wrappers import FrameStack
from natsort import natsorted

from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3 import PPO, DQN, A2C

# our imports
from environments.factory.factory_dirt import DirtFactory, DirtProperties
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

    run_id = '1623923982'
    model_name = 'A2C'

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
        dirt_props = DirtProperties(clean_amount=3, gain_amount=0.2, max_global_amount=30,
                                    max_local_amount=3, spawn_frequency=1, max_spawn_ratio=0.05)
        # env_kwargs.update(n_agents=1, dirt_properties=dirt_props)
        env = DirtFactory(**env_kwargs)

        env = FrameStack(env, 4)

        exp_out_path = model_path / 'exp'
        callbacks = CallbackList(
            [MonitorCallback(filepath=exp_out_path / f'future_exp_name', plotting=True)]
        )

        n_actions = env.action_space.n

        for epoch in range(100):
            observations = env.reset()
            if render:
                if isinstance(env, FrameStack):
                    env.env.render()
                else:
                    env.render()
            done_bool = False
            r = 0
            while not done_bool:
                if env.n_agents > 1:
                    actions = [model.predict(obs, deterministic=False)[0] for obs in observations]
                else:
                    actions = model.predict(observations, deterministic=False)[0]

                observations, r, done_bool, info_obj = env.step(actions)
                if render:
                    env.render()
                if done_bool:
                    break
            print(f'Factory run {epoch} done, reward is:\n    {r}')

    if out_path:
        combine_runs(out_path.parent)
