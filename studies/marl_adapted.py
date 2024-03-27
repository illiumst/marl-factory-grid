from pathlib import Path

from marl_factory_grid.algorithms.marl.iac import LoopIAC
from marl_factory_grid.algorithms.utils import load_yaml_file

if __name__ == '__main__':
    cfg_path = Path('../marl_factory_grid/algorithms/marl/example_config.yaml')

    cfg = load_yaml_file(cfg_path)

    print("Training phase")
    agent = LoopIAC(cfg)
    agent.train_loop()
    print("Evaluation phase")
    agent.eval_loop(10)