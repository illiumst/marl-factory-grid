from pathlib import Path
from marl_factory_grid.algorithms.marl.a2c_coin import A2C
from marl_factory_grid.algorithms.marl.utils import get_algorithms_marl_path
from marl_factory_grid.algorithms.utils import load_yaml_file


####### Training routines ######
def rerun_coin_quadrant_agent1_training():
    train_cfg_path = Path(f'./marl_factory_grid/algorithms/marl/single_agent_configs/coin_quadrant_train_config.yaml')
    eval_cfg_path = Path(f'./marl_factory_grid/algorithms/marl/single_agent_configs/coin_quadrant_eval_config.yaml')
    train_cfg = load_yaml_file(train_cfg_path)
    eval_cfg = load_yaml_file(eval_cfg_path)

    print("Training phase")
    agent = A2C(train_cfg=train_cfg, eval_cfg=eval_cfg, mode="train")
    agent.train_loop()
    print("Evaluation phase")
    agent.eval_loop("coin_quadrant", n_episodes=1)


def two_rooms_training(max_steps, agent_name):
    train_cfg_path = Path(f'./marl_factory_grid/algorithms/marl/single_agent_configs/two_rooms_train_config.yaml')
    eval_cfg_path = Path(f'./marl_factory_grid/algorithms/marl/single_agent_configs/two_rooms_eval_config.yaml')
    train_cfg = load_yaml_file(train_cfg_path)
    eval_cfg = load_yaml_file(eval_cfg_path)

    # train_cfg["algorithm"]["max_steps"] = max_steps
    train_cfg["env"]["env_name"] = f"marl/single_agent_configs/two_rooms_{agent_name}_train_config"
    eval_cfg["env"]["env_name"] = f"marl/single_agent_configs/two_rooms_{agent_name}_eval_config"
    print("Training phase")
    agent = A2C(train_cfg=train_cfg, eval_cfg=eval_cfg, mode="train")
    agent.train_loop()
    print("Evaluation phase")
    agent.eval_loop("two_rooms", n_episodes=1)


def rerun_two_rooms_agent1_training():
    two_rooms_training(max_steps=190000, agent_name="agent1")


def rerun_two_rooms_agent2_training():
    two_rooms_training(max_steps=260000, agent_name="agent2")


####### Eval routines ########
def single_agent_eval(config_name, run_folder_name):
    eval_cfg_path = Path(f'./marl_factory_grid/algorithms/marl/single_agent_configs/{config_name}_eval_config.yaml')
    eval_cfg = load_yaml_file(eval_cfg_path)

    # A value for train_cfg is required, but the train environment won't be used
    agent = A2C(eval_cfg=eval_cfg, mode="eval")
    print("Evaluation phase")
    agent.load_agents(config_name, [run_folder_name])
    agent.eval_loop(config_name, 1)


def multi_agent_eval(config_name, runs, emergent_phenomenon=False):
    eval_cfg_path = Path(f'{get_algorithms_marl_path()}/multi_agent_configs/{config_name}' +
                         f'_eval_config{"_emergent" if emergent_phenomenon else ""}.yaml')
    eval_cfg = load_yaml_file(eval_cfg_path)

    # A value for train_cfg is required, but the train environment won't be used
    agent = A2C(eval_cfg=eval_cfg, mode="eval")
    print("Evaluation phase")
    agent.load_agents(config_name, runs)
    agent.eval_loop(config_name, 1)


def coin_quadrant_multi_agent_rl_eval(emergent_phenomenon):
    # Using an empty list for runs indicates, that the default agents in algorithms/agent_models should be used.
    # If you want to use different agents, that were obtained by running the training with a different seed, you can
    # load these agents by inserting the names of the runs in study_out/ into the runs list e.g. ["run1", "run2"]
    multi_agent_eval("coin_quadrant", [], emergent_phenomenon)


def two_rooms_multi_agent_rl_eval(emergent_phenomenon):
    # Using an empty list for runs indicates, that the default agents in algorithms/agent_models should be used.
    # If you want to use different agents, that were obtained by running the training with a different seed, you can
    # load these agents by inserting the names of the runs in study_out/ into the runs list e.g. ["run1", "run2"]
    multi_agent_eval("two_rooms", [], emergent_phenomenon)
