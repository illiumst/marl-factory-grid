from pathlib import Path
from marl_factory_grid.algorithms.rl.a2c_dirt import A2C
from marl_factory_grid.algorithms.utils import load_yaml_file


####### Training routines ######
def rerun_dirt_quadrant_agent1_training():
    train_cfg_path = Path(f'./marl_factory_grid/algorithms/rl/single_agent_configs/dirt_quadrant_train_config.yaml')
    eval_cfg_path = Path(f'./marl_factory_grid/algorithms/rl/single_agent_configs/dirt_quadrant_eval_config.yaml')
    train_cfg = load_yaml_file(train_cfg_path)
    eval_cfg = load_yaml_file(eval_cfg_path)

    print("Training phase")
    agent = A2C(train_cfg, eval_cfg)
    agent.train_loop()
    print("Evaluation phase")
    agent.eval_loop(n_episodes=1)


def two_rooms_training(max_steps, agent_name):
    train_cfg_path = Path(f'./marl_factory_grid/algorithms/rl/single_agent_configs/two_rooms_train_config.yaml')
    eval_cfg_path = Path(f'./marl_factory_grid/algorithms/rl/single_agent_configs/two_rooms_eval_config.yaml')
    train_cfg = load_yaml_file(train_cfg_path)
    eval_cfg = load_yaml_file(eval_cfg_path)

    train_cfg["algorithm"]["max_steps"] = max_steps
    train_cfg["env"]["env_name"] = f"rl/two_rooms_{agent_name}_train_config"
    eval_cfg["env"]["env_name"] = f"rl/two_rooms_{agent_name}_eval_config"
    print("Training phase")
    agent = A2C(train_cfg, eval_cfg)
    agent.train_loop()
    print("Evaluation phase")
    agent.eval_loop(n_episodes=1)


def rerun_two_rooms_agent1_training():
    two_rooms_training(max_steps=190000, agent_name="agent1")


def rerun_two_rooms_agent2_training():
    two_rooms_training(max_steps=260000, agent_name="agent2")


####### Eval routines ########
def single_agent_eval(config_name, run_folder_name):
    eval_cfg_path = Path(f'../marl_factory_grid/algorithms/rl/single_agent_configs/{config_name}_eval_config.yaml')
    train_cfg = eval_cfg = load_yaml_file(eval_cfg_path)

    # A value for train_cfg is required, but the train environment won't be used
    agent = A2C(train_cfg=train_cfg, eval_cfg=eval_cfg)
    print("Evaluation phase")
    agent.load_agents([run_folder_name])
    agent.eval_loop(1)


def multi_agent_eval(config_name, runs, emergent_phenomenon=False):
    eval_cfg_path = Path(f'./marl_factory_grid/algorithms/rl/multi_agent_configs/{config_name}' +
                         f'_eval_config{"_emergent" if emergent_phenomenon else ""}.yaml')
    eval_cfg = load_yaml_file(eval_cfg_path)

    # A value for train_cfg is required, but the train environment won't be used
    agent = A2C(train_cfg=eval_cfg, eval_cfg=eval_cfg)
    print("Evaluation phase")
    agent.load_agents(runs)
    agent.eval_loop(1)


def dirt_quadrant_multi_agent_rl_eval(emergent_phenomenon):
    multi_agent_eval("dirt_quadrant", ["run0", "run0"], emergent_phenomenon)


def two_rooms_multi_agent_rl_eval(emergent_phenomenon):
    multi_agent_eval("two_rooms", ["run1", "run2"], emergent_phenomenon)