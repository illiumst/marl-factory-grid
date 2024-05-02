import copy
from pathlib import Path
from marl_factory_grid.algorithms.marl.a2c_dirt import A2C
from marl_factory_grid.algorithms.utils import load_yaml_file

if __name__ == '__main__':
    cfg_path = Path('../marl_factory_grid/algorithms/marl/configs/dirt_quadrant_config.yaml')

    train_cfg = load_yaml_file(cfg_path)
    # Use environment config with fixed spawnpoints for eval
    eval_cfg = copy.deepcopy(train_cfg)
    eval_cfg["env"]["env_name"] = "custom/dirt_quadrant" # Options: two_rooms_one_door_modified, dirt_quadrant

    print("Training phase")
    agent = A2C(train_cfg, eval_cfg)
    agent.train_loop()
    agent.plot_reward_development()
    print("Evaluation phase")
    agent.eval_loop(10)