import ast

from os import PathLike
from pathlib import Path
from typing import Union, List

import yaml

from marl_factory_grid.environment.rules import Rule
from marl_factory_grid.utils.helpers import locate_and_import_class
from marl_factory_grid.environment.constants import DEFAULT_PATH, MODULE_PATH
from marl_factory_grid.environment import constants as c


class FactoryConfigParser(object):
    default_entites = []
    default_rules = ['DoneAtMaxStepsReached', 'WatchCollision']
    default_actions = [c.MOVE8, c.NOOP]
    default_observations = [c.WALLS, c.AGENT]

    def __init__(self, config_path, custom_modules_path: Union[PathLike] = None):
        self.config_path = Path(config_path)
        self.custom_modules_path = Path(custom_modules_path) if custom_modules_path is not None else custom_modules_path
        self.config = yaml.safe_load(self.config_path.open())

    def __getattr__(self, item):
        return self['General'][item]

    def _get_sub_list(self, primary_key: str, sub_key: str):
        return [{key: [s for k, v in val.items() if k == sub_key for s in v] for key, val in x.items()
                 } for x in self.config[primary_key]]

    @property
    def agent_actions(self):
        return self._get_sub_list('Agents', "Actions")

    @property
    def agent_observations(self):
        return self._get_sub_list('Agents', "Observations")

    @property
    def rules(self):
        return self.config['Rules']

    @property
    def tests(self):
        return self.config.get('Tests', [])

    @property
    def agents(self):
        return self.config['Agents']

    @property
    def entities(self):
        return self.config['Entities']

    def __repr__(self):
        return str(self.config)

    def __getitem__(self, item):
        return self.config[item]

    def load_entities(self):
        entity_classes = dict()
        entities = []
        if c.DEFAULTS in self.entities:
            entities.extend(self.default_entites)
        entities.extend(x for x in self.entities if x != c.DEFAULTS)

        for entity in entities:
            e1 = e2 = e3 = None
            try:
                folder_path = Path(__file__).parent.parent / DEFAULT_PATH
                entity_class = locate_and_import_class(entity, folder_path)
            except AttributeError as e:
                e1 = e
                try:
                    module_path = Path(__file__).parent.parent / MODULE_PATH
                    entity_class = locate_and_import_class(entity, module_path)
                except AttributeError as e:
                    e2 = e
                    if self.custom_modules_path:
                        try:
                            entity_class = locate_and_import_class(entity, self.custom_modules_path)
                        except AttributeError as e:
                            e3 = e
                            pass
            if (e1 and e2) or e3:
                ents = [y for x in [e1, e2, e3] if x is not None for y in x.args[1]]
                print('##############################################################')
                print('### Error  ###  Error  ###  Error  ###  Error  ###  Error  ###')
                print('##############################################################')
                print(f'Class "{entity}" was not found in "{module_path.name}"')
                print(f'Class "{entity}" was not found in "{folder_path.name}"')
                print('##############################################################')
                if self.custom_modules_path:
                    print(f'Class "{entity}" was not found in "{self.custom_modules_path}"')
                print('Possible Entitys are:', str(ents))
                print('##############################################################')
                print('Goodbye')
                print('##############################################################')
                print('### Error  ###  Error  ###  Error  ###  Error  ###  Error  ###')
                print('##############################################################')
                exit(-99999)

            entity_kwargs = self.entities.get(entity, {})
            entity_symbol = entity_class.symbol if hasattr(entity_class, 'symbol') else None
            entity_classes.update({entity: {'class': entity_class, 'kwargs': entity_kwargs, 'symbol': entity_symbol}})
        return entity_classes

    def parse_agents_conf(self):
        parsed_agents_conf = dict()
        base_env_actions  = self.default_actions.copy() + [c.MOVE4]
        for name in self.agents:
            # Actions
            actions = list()
            if c.DEFAULTS in self.agents[name]['Actions']:
                actions.extend(self.default_actions)
            actions.extend(x for x in self.agents[name]['Actions'] if x != c.DEFAULTS)
            parsed_actions = list()
            for action in actions:
                folder_path = MODULE_PATH if action not in base_env_actions else DEFAULT_PATH
                folder_path = Path(__file__).parent.parent / folder_path
                try:
                    class_or_classes = locate_and_import_class(action, folder_path)
                except AttributeError:
                    class_or_classes = locate_and_import_class(action, self.custom_modules_path)
                try:
                    parsed_actions.extend(class_or_classes)
                except TypeError:
                    parsed_actions.append(class_or_classes)

            parsed_actions = [x() for x in parsed_actions]

            # Observation
            observations = list()
            assert self.agents[name]['Observations'] is not None, 'Did you specify any Observation?'
            if c.DEFAULTS in self.agents[name]['Observations']:
                observations.extend(self.default_observations)
            observations.extend(x for x in self.agents[name]['Observations'] if x != c.DEFAULTS)
            positions = [ast.literal_eval(x) for x in self.agents[name].get('Positions', [])]
            other_kwargs = {k: v for k, v in self.agents[name].items() if k not in
                            ['Actions', 'Observations', 'Positions']}
            parsed_agents_conf[name] = dict(
                actions=parsed_actions, observations=observations, positions=positions, other=other_kwargs
                                            )

        return parsed_agents_conf

    def load_env_rules(self) -> List[Rule]:
        rules = self.rules.copy()
        if c.DEFAULTS in self.rules:
            for rule in self.default_rules:
                if rule not in rules:
                    rules.append({rule: {}})

        return self._load_smth(rules, Rule)

    def load_env_tests(self) -> List[Rule]:
        return self._load_smth(self.tests, None)  # Test

    def _load_smth(self, config, class_obj):
        rules = list()
        rules_names = list()
        for rule in config:
            e1 = e2 = e3 = None
            try:
                folder_path = (Path(__file__).parent.parent / DEFAULT_PATH)
                rule_class = locate_and_import_class(rule, folder_path)
            except AttributeError as e:
                e1 = e
                try:
                    module_path = (Path(__file__).parent.parent / MODULE_PATH)
                    rule_class = locate_and_import_class(rule, module_path)
                except AttributeError as e:
                    e2 = e
                    if self.custom_modules_path:
                        try:
                            rule_class = locate_and_import_class(rule, self.custom_modules_path)
                        except AttributeError as e:
                            e3 = e
                            pass
            if (e1 and e2) or e3:
                ents = [y for x in [e1, e2, e3] if x is not None for y in x.args[1]]
                print('### Error  ###  Error  ###  Error  ###  Error  ###  Error  ###')
                print('')
                print(f'Class "{rule}" was not found in "{module_path.name}"')
                print(f'Class "{rule}" was not found in "{folder_path.name}"')
                if self.custom_modules_path:
                    print(f'Class "{rule}" was not found in "{self.custom_modules_path}"')
                print('Possible Entitys are:', str(ents))
                print('')
                print('Goodbye')
                print('')
                exit(-99999)

            if issubclass(rule_class, class_obj):
                rule_kwargs = config.get(rule, {})
                rules.append(rule_class(**(rule_kwargs or {})))
        return rules

    def load_entity_spawn_rules(self, entities) -> List[Rule]:
        rules = list()
        rules_dicts = list()
        for e in entities:
            try:
                if spawn_rule := e.spawn_rule:
                    rules_dicts.append(spawn_rule)
            except AttributeError:
                pass

        for rule_dict in rules_dicts:
            for rule_name, rule_kwargs in rule_dict.items():
                try:
                    folder_path = (Path(__file__).parent.parent / DEFAULT_PATH)
                    rule_class = locate_and_import_class(rule_name, folder_path)
                except AttributeError:
                    try:
                        folder_path = (Path(__file__).parent.parent / MODULE_PATH)
                        rule_class = locate_and_import_class(rule_name, folder_path)
                    except AttributeError:
                        rule_class = locate_and_import_class(rule_name, self.custom_modules_path)
                rules.append(rule_class(**rule_kwargs))
        return rules
