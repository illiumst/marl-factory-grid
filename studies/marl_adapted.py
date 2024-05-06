import copy
from pathlib import Path
from marl_factory_grid.algorithms.marl.a2c_dirt import A2C
from marl_factory_grid.algorithms.utils import load_yaml_file

def dirt_quadrant_single_agent_training():
    cfg_path = Path('../marl_factory_grid/algorithms/marl/configs/dirt_quadrant_config.yaml')

    train_cfg = load_yaml_file(cfg_path)
    # Use environment config with fixed spawnpoints for eval
    eval_cfg = copy.deepcopy(train_cfg)
    eval_cfg["env"]["env_name"] = "custom/dirt_quadrant_eval_config"

    print("Training phase")
    agent = A2C(train_cfg, eval_cfg)
    agent.train_loop()
    print("Evaluation phase")
    # Have consecutive episode for eval in single agent case
    train_cfg["algorithm"]["pile_all_done"] = "all"
    # agent.load_agents(["run0", "run1"])
    agent.eval_loop(10)


def dirt_quadrant_multi_agent_eval():
    cfg_path = Path('../marl_factory_grid/algorithms/marl/configs/MultiAgentConfigs/dirt_quadrant_config.yaml')

    train_cfg = load_yaml_file(cfg_path)
    # Use environment config with fixed spawnpoints for eval
    eval_cfg = copy.deepcopy(train_cfg)
    eval_cfg["env"]["env_name"] = "custom/MultiAgentConfigs/dirt_quadrant_eval_config"
    agent = A2C(train_cfg, eval_cfg)
    print("Evaluation phase")
    agent.load_agents(["run0", "run1"])
    agent.eval_loop(10)


if __name__ == '__main__':
    dirt_quadrant_single_agent_training()