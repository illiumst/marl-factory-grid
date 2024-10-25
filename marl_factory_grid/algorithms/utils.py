import os
from pathlib import Path

import numpy as np
import yaml

from marl_factory_grid import Factory
from marl_factory_grid.algorithms.marl.utils import get_configs_marl_path


def load_class(classname):
    from importlib import import_module
    module_path, class_name = classname.rsplit(".", 1)
    module = import_module(module_path)
    c = getattr(module, class_name)
    return c


def instantiate_class(arguments):
    from importlib import import_module

    d = dict(arguments)
    classname = d["classname"]
    del d["classname"]
    module_path, class_name = classname.rsplit(".", 1)
    module = import_module(module_path)
    c = getattr(module, class_name)
    return c(**d)


def get_class(arguments):
    from importlib import import_module

    if isinstance(arguments, dict):
        classname = arguments["classname"]
        module_path, class_name = classname.rsplit(".", 1)
        module = import_module(module_path)
        c = getattr(module, class_name)
        return c
    else:
        classname = arguments.classname
        module_path, class_name = classname.rsplit(".", 1)
        module = import_module(module_path)
        c = getattr(module, class_name)
        return c


def get_study_out_path():
    return Path(os.path.join(Path(__file__).parent.parent.parent, "study_out"))


def get_arguments(arguments):
    d = dict(arguments)
    if "classname" in d:
        del d["classname"]
    return d


def load_yaml_file(path: Path):
    with path.open() as stream:
        cfg = yaml.load(stream, Loader=yaml.FullLoader)
    return cfg


def add_env_props(cfg):
    # Path to config File
    env_path = Path(f'{get_configs_marl_path()}/{cfg["env"]["env_name"]}.yaml')
    print(cfg)

    # Env Init
    factory = Factory(env_path)
    _ = factory.reset()

    return factory


class Checkpointer(object):
    def __init__(self, experiment_name, root, config, total_steps, n_checkpoints):
        self.path = root / experiment_name
        self.checkpoint_indices = list(np.linspace(1, total_steps, n_checkpoints, dtype=int) - 1)
        self.__current_checkpoint = 0
        self.__current_step = 0
        self.path.mkdir(exist_ok=True, parents=True)
        with (self.path / 'config.yaml').open('w') as outfile:
            yaml.dump(config, outfile, default_flow_style=False)

    def save_experiment(self, name: str, model):
        cpt_path = self.path / f'checkpoint_{self.__current_checkpoint}'
        cpt_path.mkdir(exist_ok=True, parents=True)
        import torch
        torch.save(model.state_dict(), cpt_path / f'{name}.pt')

    def step(self, to_save):
        if self.__current_step in self.checkpoint_indices:
            print(f'Checkpointing #{self.__current_checkpoint}')
            for name, model in to_save:
                self.save_experiment(name, model)
            self.__current_checkpoint += 1
        self.__current_step += 1
