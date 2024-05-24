import copy
from pathlib import Path
from marl_factory_grid.algorithms.marl.a2c_dirt import A2C
from marl_factory_grid.algorithms.utils import load_yaml_file

def single_agent_training(config_name):
    cfg_path = Path(f'../marl_factory_grid/algorithms/marl/single_agent_configs/{config_name}_config.yaml')

    train_cfg = load_yaml_file(cfg_path)
    # Use environment config with fixed spawnpoints for eval
    eval_cfg = copy.deepcopy(train_cfg)
    eval_cfg["env"]["env_name"] = f"rl/{config_name}_eval_config"

    print("Training phase")
    agent = A2C(train_cfg, eval_cfg)
    agent.train_loop()
    print("Evaluation phase")
    # Have consecutive episode for eval in single agent case
    train_cfg["algorithm"]["pile_all_done"] = "all"
    agent.eval_loop(10)


def single_agent_eval(config_name, run):
    cfg_path = Path(f'../marl_factory_grid/algorithms/marl/single_agent_configs/{config_name}_config.yaml')

    train_cfg = load_yaml_file(cfg_path)
    # Use environment config with fixed spawnpoints for eval
    eval_cfg = copy.deepcopy(train_cfg)
    eval_cfg["env"]["env_name"] = f"rl/{config_name}_eval_config"
    agent = A2C(train_cfg, eval_cfg)
    print("Evaluation phase")
    agent.load_agents(run)
    agent.eval_loop(1)


def multi_agent_eval(config_name, runs, emergent_phenomenon=False):
    cfg_path = Path(f'../marl_factory_grid/algorithms/marl/multi_agent_configs/{config_name}_config.yaml')

    eval_cfg = load_yaml_file(cfg_path)
    #  Sanity setting of required attributes and configs
    if config_name == "two_rooms":
        if emergent_phenomenon:
            eval_cfg["env"]["env_name"] = f"marl_eval/{config_name}_eval_config_emergent"
            eval_cfg["algorithm"]["auxiliary_piles"] = False
        else:
            eval_cfg["algorithm"]["auxiliary_piles"] = True
    elif config_name == "dirt_quadrant":
        if emergent_phenomenon:
            eval_cfg["algorithm"]["pile-order"] = "dynamic"
        else:
            eval_cfg["algorithm"]["pile-order"] = "smart"
    agent = A2C(train_cfg=eval_cfg, eval_cfg=eval_cfg)
    print("Evaluation phase")
    agent.load_agents(runs)
    agent.eval_loop(1)


def dirt_quadrant_single_agent_training():
    single_agent_training("dirt_quadrant")


def two_rooms_one_door_modified_single_agent_training():
    single_agent_training("two_rooms")


def dirt_quadrant_single_agent_eval(agent_name):
    if agent_name == "Sigmund":
        run = "run0"
    elif agent_name == "Wolfgang":
        run = "run1"
    single_agent_eval("dirt_quadrant", [run])


def two_rooms_one_door_modified_single_agent_eval(agent_name):
    if agent_name == "Sigmund":
        run = "run2"
    elif agent_name == "Wolfgang":
        run = "run3"
    single_agent_eval("two_rooms", [run])


def dirt_quadrant_5_multi_agent_eval(emergent_phenomenon):
    multi_agent_eval("dirt_quadrant", ["run4", "run5"], emergent_phenomenon)

def dirt_quadrant_5_multi_agent_ctde_eval(emergent_phenomenon): # run7 == run4
    multi_agent_eval("dirt_quadrant", ["run4", "run7"], emergent_phenomenon)

def two_rooms_one_door_modified_multi_agent_eval(emergent_phenomenon):
    multi_agent_eval("two_rooms", ["run2", "run3"], emergent_phenomenon)


if __name__ == '__main__':
    two_rooms_one_door_modified_multi_agent_eval(False)