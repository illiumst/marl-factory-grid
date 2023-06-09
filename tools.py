import importlib
import inspect
from os import PathLike
from pathlib import Path

import yaml

from environment import constants as c
from environment.utils.helpers import locate_and_import_class

ACTION       = 'Action'
GENERAL      = 'General'
ENTITIES     = 'Objects'
OBSERVATIONS = 'Observations'
RULES        = 'Rule'
ASSETS       = 'Assets'
EXCLUDED     = ['identifier', 'args', 'kwargs', 'Move',
                'TemplateRule', 'Entities', 'EnvObjects', 'Zones', ]


class ConfigExplainer:

    def __init__(self, base_path: PathLike = 'environment',
                 module_path: PathLike = 'modules'):
        self.module_path = Path(module_path)
        self.base_path = Path(base_path)
        self.searchspace = [ACTION, GENERAL, ENTITIES, OBSERVATIONS, RULES, ASSETS]

    def explain_module(self, class_to_explain):
        parameters = inspect.signature(class_to_explain).parameters
        explained = {class_to_explain.__name__: {key: val.default for key, val in parameters.items() if key not in EXCLUDED}}
        return explained

    def _load_and_compare(self, compare_class, paths):
        conf = {}
        for module_path in paths:
            mods = importlib.import_module('.'.join([x.replace('.py', '') for x in module_path.parts]))
            for key in mods.__dict__.keys():
                if key not in EXCLUDED and not key.startswith('_'):
                    mod = mods.__getattribute__(key)
                    try:
                        if issubclass(mod, compare_class) and mod != compare_class:
                            conf.update(self.explain_module(mod))
                    except TypeError:
                        pass
        return conf

    def save_actions(self, output_conf_file: PathLike = Path('quickstart') / 'explained_actions.yml'):
        self._save_to_file(self.get_entities(), output_conf_file, ACTION)

    def get_actions(self):
        actions = self._get_by_identifier(ACTION)
        assert all(not x for x in actions.values()), 'Please only provide Names, no Mappings.'
        actions = list(actions.keys())
        actions.extend([c.MOVE8, c.MOVE4])
        # TODO: Print to file!
        return actions

    def save_entities(self, output_conf_file: PathLike = Path('quickstart') / 'explained_entities.yml'):
        self._save_to_file(self.get_entities(), output_conf_file, ENTITIES)

    def get_entities(self):
        entities = self._get_by_identifier(ENTITIES)
        return entities

    def save_rules(self, output_conf_file: PathLike = Path('quickstart') / 'explained_rules.yml'):
        self._save_to_file(self.get_entities(), output_conf_file, RULES)

    def get_rules(self):
        rules = self._get_by_identifier(RULES)
        return rules

    def get_assets(self):
        pass

    def get_observations(self):
        names = [c.ALL, c.COMBINED, c.SELF, c.OTHERS, "Agent['ExampleAgentName']"]
        for key, val in self.get_entities().items():
            try:
                e = locate_and_import_class(key)(level_shape=(0, 0), pomdp_r=0).obs_pairs
            except TypeError:
                e = [key]
            names.extend(e)
        return names

    def _get_by_identifier(self, identifier):
        entities_base_cls = locate_and_import_class(identifier, self.base_path)
        module_paths = [x for x in self.base_path.rglob('*.py') if x.is_file() and '__init__' not in x.name]
        found_entities = self._load_and_compare(entities_base_cls, module_paths)
        module_paths = [x for x in self.module_path.rglob('*.py') if x.is_file() and '__init__' not in x.name]
        found_entities.update(self._load_and_compare(entities_base_cls, module_paths))
        return found_entities

    def save_all(self, output_conf_file: PathLike = Path('quickstart') / 'explained.yml'):
        self._save_to_file(self.get_all(), output_conf_file, 'ALL')

    def get_all(self):
        config_dict = {GENERAL: {'level_name': 'rooms', 'env_seed': 69, 'verbose': False,
                                 'pomdp_r': 3, 'individual_rewards': True},
                       'Agents': dict(
                           ExampleAgentName=dict(
                               Actions=self.get_actions(),
                               Observations=self.get_observations())),
                       'Entities': self.get_entities(),
                       'Rules': self.get_rules(),
                       'Assets': self.get_assets()}
        return config_dict

    def _save_to_file(self, data: dict, filepath: PathLike, tag: str = ''):
        filepath = Path(filepath)
        with filepath.open('w') as f:
            yaml.dump(data, f, encoding='utf-8')
        print(f'Example config {"for " + tag + " " if tag else " "}dumped')
        print(f'See file: {filepath}')


if __name__ == '__main__':
    ce = ConfigExplainer()
    ce.get_actions()
    ce.get_entities()
    ce.get_rules()
    ce.get_observations()
    ce.get_assets()
    all_conf = ce.get_all()
    print()
