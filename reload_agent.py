import warnings
from pathlib import Path

import yaml
from stable_baselines3 import A2C, PPO, DQN

from environments.factory.factory_dirt import DirtFactory
from environments.logging.recorder import EnvRecorder

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)


if __name__ == '__main__':

    determin = False
    render = True
    record = False
    seed = 67
    n_agents = 1
    # out_path = Path('study_out/e_1_new_reward/no_obs/dirt/A2C_new_reward/0_A2C_new_reward')
    out_path = Path('study_out/test/dirt')
    model_path = out_path

    with (out_path / f'env_params.json').open('r') as f:
        env_kwargs = yaml.load(f, Loader=yaml.FullLoader)
        env_kwargs.update(additional_agent_placeholder=None, n_agents=n_agents, max_steps=150)
        if gain_amount := env_kwargs.get('dirt_prop', {}).get('gain_amount', None):
            env_kwargs['dirt_prop']['max_spawn_amount'] = gain_amount
            del env_kwargs['dirt_prop']['gain_amount']

        env_kwargs.update(record_episodes=record, done_at_collision=True)

    this_model = out_path / 'model.zip'

    model_cls = PPO  # next(val for key, val in h.MODEL_MAP.items() if key in out_path.parent.name)
    models = [model_cls.load(this_model)]

    # Init Env
    with DirtFactory(**env_kwargs) as env:
        env = EnvRecorder(env)
        obs_shape = env.observation_space.shape
        # Evaluation Loop for i in range(n Episodes)
        for episode in range(50):
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
                if done_bool:
                    break
            print(f'Factory run {episode} done, reward is:\n    {rew}')
    print('all done')
