import importlib
import inspect
from os import PathLike
from pathlib import Path

import yaml

from marl_factory_grid.environment import constants as c
from marl_factory_grid.utils.helpers import locate_and_import_class

ACTION = 'Action'
GENERAL = 'General'
ENTITIES = 'Objects'
OBSERVATIONS = 'Observations'
RULES = 'Rule'
TESTS = 'Tests'
EXCLUDED = ['identifier', 'args', 'kwargs', 'Move', 'Agent', 'GlobalPositions', 'Walls',
            'TemplateRule', 'Entities', 'EnvObjects', 'Zones', ]


class ConfigExplainer:

    def __init__(self, custom_path: None | PathLike = None):
        """
        This utility serves as a helper for debugging and exploring available modules and classes.
        Does not do anything unless told.
        The functions get_xxxxx() retrieves and returns the information and save_xxxxx() dumps them to disk.

        get_all() and save_all() helps geting a general overview.

        When provided with a custom path, your own modules become available.

        :param custom_path: Path to your custom module folder.
        """
        self.base_path = Path(__file__).parent.parent.resolve()
        self.custom_path = Path(custom_path) if custom_path is not None else custom_path
        self.searchspace = [ACTION, GENERAL, ENTITIES, OBSERVATIONS, RULES, TESTS]

    @staticmethod
    def _explain_module(class_to_explain):
        """
        INTERNAL USE ONLY
        """
        parameters = inspect.signature(class_to_explain).parameters
        explained = {class_to_explain.__name__:
                         {key: val.default for key, val in parameters.items() if key not in EXCLUDED}
                     }
        return explained

    def _get_by_identifier(self, identifier):
        """
        INTERNAL USE ONLY
        """
        entities_base_cls = locate_and_import_class(identifier, self.base_path)
        module_paths = [x.resolve() for x in self.base_path.rglob('*.py') if x.is_file() and '__init__' not in x.name]
        found_entities = self._load_and_compare(entities_base_cls, module_paths)
        if self.custom_path is not None:
            module_paths = [x.resolve() for x in self.custom_path.rglob('*.py') if x.is_file()
                            and '__init__' not in x.name]
            found_entities.update(self._load_and_compare(entities_base_cls, module_paths))
        return found_entities

    def _load_and_compare(self, compare_class, paths):
        """
        INTERNAL USE ONLY
        """
        conf = {}
        package_pos = next(idx for idx, x in enumerate(Path(__file__).resolve().parts) if x == 'marl_factory_grid')
        for module_path in paths:
            module_parts = [x.replace('.py', '') for idx, x in enumerate(module_path.parts) if idx >= package_pos]
            mods = importlib.import_module('.'.join(module_parts))
            for key in mods.__dict__.keys():
                if key not in EXCLUDED and not key.startswith('_'):
                    mod = mods.__getattribute__(key)
                    try:
                        if issubclass(mod, compare_class) and mod != compare_class:
                            conf.update(self._explain_module(mod))
                    except TypeError:
                        pass
        return conf

    @staticmethod
    def _save_to_file(data: dict, filepath: PathLike, tag: str = ''):
        """
        INTERNAL USE ONLY
        """
        filepath = Path(filepath)
        yaml.Dumper.ignore_aliases = lambda *args: True
        with filepath.open('w') as f:
            yaml.dump(data, f, encoding='utf-8')
        print(f'Example config {"for " + tag + " " if tag else " "}dumped')
        print(f'See file: {filepath}')

    def get_actions(self) -> list[str]:
        """
        Retrieve all actions from module folders.

        :returns: A list of all available actions.
        """
        actions = self._get_by_identifier(ACTION)
        assert all(not x for x in actions.values()), 'Please only provide Names, no Mappings.'
        actions = list(actions.keys())
        actions.extend([c.MOVE8, c.MOVE4])
        return actions

    def get_all(self) -> dict[str]:
        """
        Retrieve all available configurations from module folders.

        :returns: A dictionary of all available configurations.
        """

        config_dict = {
            'General': self.get_general_section(),
            'Agents': self.get_agent_section(),
            'Entities': self.get_entities(),
            'Rules': self.get_rules()
        }
        return config_dict

    def get_entities(self):
        """
        Retrieve all entities from module folders.

        :returns: A list of all available entities.
        """
        entities = self._get_by_identifier(ENTITIES)
        return entities

    @staticmethod
    def get_general_section():
        """
        Build the general section.

        :returns: A list of all available entities.
        """
        general = {'level_name': 'rooms', 'env_seed': 69, 'verbose': False,
                   'pomdp_r': 3, 'individual_rewards': True, 'tests': False}
        return general

    def get_agent_section(self):
        """
        Build the Agent section and retrieve all available actions and observations from module folders.

        :returns: Agent section.
        """
        agents = dict(
            ExampleAgentName=dict(
                Actions=self.get_actions(),
                Observations=self.get_observations())),
        return agents

    def get_rules(self) -> dict[str]:
        """
        Retrieve all rules from module folders.

        :returns: All available rules.
        """
        rules = self._get_by_identifier(RULES)
        return rules

    def get_observations(self) -> list[str]:
        """
        Retrieve all agent observations from module folders.

        :returns: A list of all available observations.
        """
        names = [c.ALL, c.COMBINED, c.SELF, c.OTHERS, "Agent['ExampleAgentName']"]
        for key, val in self.get_entities().items():
            try:
                e = locate_and_import_class(key, self.base_path)(level_shape=(0, 0), pomdp_r=0).obs_pairs
            except TypeError:
                e = [key]
            except AttributeError as err:
                if self.custom_path is not None:
                    try:
                        e = locate_and_import_class(key, self.base_path)(level_shape=(0, 0), pomdp_r=0).obs_pairs
                    except TypeError:
                        e = [key]
                else:
                    raise err
            names.extend(e)
        return names

    def save_actions(self, output_conf_file: PathLike = Path('../../quickstart') / 'actions.yml'):
        """
        Write all availale actions to a file.
        :param output_conf_file: File to write to. Defaults to ../../quickstart/actions.yml
        """
        self._save_to_file(self.get_entities(), output_conf_file, ACTION)

    def save_entities(self, output_conf_file: PathLike = Path('../../quickstart') / 'entities.yml'):
        """
        Write all availale entities to a file.
        :param output_conf_file: File to write to. Defaults to ../../quickstart/entities.yml
        """
        self._save_to_file(self.get_entities(), output_conf_file, ENTITIES)

    def save_observations(self, output_conf_file: PathLike = Path('../../quickstart') / 'observations.yml'):
        """
        Write all availale observations to a file.
        :param output_conf_file: File to write to. Defaults to ../../quickstart/observations.yml
        """
        self._save_to_file(self.get_entities(), output_conf_file, OBSERVATIONS)

    def save_rules(self, output_conf_file: PathLike = Path('../../quickstart') / 'rules.yml'):
        """
        Write all availale rules to a file.
        :param output_conf_file: File to write to. Defaults to ../../quickstart/rules.yml
        """
        self._save_to_file(self.get_entities(), output_conf_file, RULES)

    def save_all(self, output_conf_file: PathLike = Path('../../quickstart') / 'all.yml'):
        """
        Write all availale keywords to a file.
        :param output_conf_file: File to write to. Defaults to ../../quickstart/all.yml
        """
        self._save_to_file(self.get_all(), output_conf_file, 'ALL')


if __name__ == '__main__':
    ce = ConfigExplainer()
    # ce.get_actions()
    # ce.get_entities()
    # ce.get_rules()
    # ce.get_observations()
    all_conf = ce.get_all()
    ce.save_all()
