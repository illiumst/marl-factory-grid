import warnings

from pathlib import Path
import time

from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.vec_env import SubprocVecEnv

from environments.factory.factory_dirt_item import DirtItemFactory
from environments.factory.factory_item import ItemFactory, ItemProperties
from environments.factory.factory_dirt import DirtProperties, DirtFactory
from environments.logging.monitor import MonitorCallback
from environments.logging.recorder import RecorderCallback
from environments.utility_classes import MovementProperties
from plotting.compare_runs import compare_seed_runs, compare_model_runs

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)


def make_env(env_kwargs_dict):

    def _init():
        with DirtFactory(**env_kwargs_dict) as init_env:
            return init_env

    return _init


if __name__ == '__main__':

    # combine_runs(Path('debug_out') / 'A2C_1630314192')
    # exit()

    # compare_runs(Path('debug_out'), 1623052687, ['step_reward'])
    # exit()

    from stable_baselines3 import PPO, DQN, A2C
    # from algorithms.reg_dqn import RegDQN
    # from sb3_contrib import QRDQN

    dirt_props = DirtProperties(clean_amount=2, gain_amount=0.1, max_global_amount=20,
                                max_local_amount=1, spawn_frequency=16, max_spawn_ratio=0.05,
                                dirt_smear_amount=0.0, agent_can_interact=True)
    item_props = ItemProperties(n_items=10, agent_can_interact=True,
                                spawn_frequency=30, n_drop_off_locations=2,
                                max_agent_inventory_capacity=15)
    move_props = MovementProperties(allow_diagonal_movement=True,
                                    allow_square_movement=True,
                                    allow_no_op=False)
    train_steps = 5e6
    time_stamp = int(time.time())

    out_path = None

    for modeL_type in [A2C, PPO, DQN]:  # ,RegDQN, QRDQN]:
        for seed in range(3):
            env_kwargs = dict(n_agents=1,
                              # item_prop=item_props,
                              dirt_properties=dirt_props,
                              movement_properties=move_props,
                              pomdp_r=2, max_steps=1000, parse_doors=False,
                              level_name='rooms', frames_to_stack=4,
                              omit_agent_in_obs=True, combin_agent_obs=True, record_episodes=False,
                              cast_shadows=True, doors_have_area=False, env_seed=seed, verbose=False,
                              )

            if modeL_type.__name__ in ["PPO", "A2C"]:
                kwargs = dict(ent_coef=0.01)
                env = SubprocVecEnv([make_env(env_kwargs) for _ in range(10)], start_method="spawn")
            elif modeL_type.__name__ in ["RegDQN", "DQN", "QRDQN"]:
                env = make_env(env_kwargs)()
                kwargs = dict(buffer_size=50000,
                              learning_starts=64,
                              batch_size=64,
                              target_update_interval=5000,
                              exploration_fraction=0.25,
                              exploration_final_eps=0.025
                              )
            else:
                raise NameError(f'The model "{modeL_type.__name__}" has the wrong name.')

            model = modeL_type("MlpPolicy", env, verbose=1, seed=seed, device='cpu', **kwargs)

            out_path = Path('debug_out') / f'{model.__class__.__name__}_{time_stamp}'

            # identifier = f'{seed}_{model.__class__.__name__}_{time_stamp}'
            identifier = f'{seed}_{model.__class__.__name__}_{time_stamp}'
            out_path /= identifier

            callbacks = CallbackList(
                [MonitorCallback(filepath=out_path / f'monitor_{identifier}.pick'),
                 RecorderCallback(filepath=out_path / f'recorder_{identifier}.json', occupation_map=False,
                                  trajectory_map=False
                                  )]
            )

            model.learn(total_timesteps=int(train_steps), callback=callbacks)

            save_path = out_path / f'model_{identifier}.zip'
            save_path.parent.mkdir(parents=True, exist_ok=True)
            model.save(save_path)
            param_path = out_path.parent / f'env_{model.__class__.__name__}_{time_stamp}.json'
            try:
                env.env_method('save_params', param_path)
            except AttributeError:
                env.save_params(param_path)
            print("Model Trained and saved")
        print("Model Group Done.. Plotting...")

        if out_path:
            compare_seed_runs(out_path.parent)
    print("All Models Done... Evaluating")
    if out_path:
        compare_model_runs(Path('debug_out'), time_stamp, 'step_reward')
