from os import PathLike
from pathlib import Path
from typing import Union

import yaml

from environment.groups.global_entities import Entities
from environment.groups.agents import Agents
from environment.entity.agent import Agent
from environment.utils.helpers import locate_and_import_class
from environment import constants as c


DEFAULT_PATH = 'environment'
MODULE_PATH = 'modules'


class FactoryConfigParser(object):

    default_entites = []
    default_rules = ['MaxStepsReached', 'Collision']
    default_actions = [c.MOVE8, c.NOOP]
    default_observations = [c.WALLS, c.AGENTS]

    def __init__(self, config_path, custom_modules_path: Union[None, PathLike] = None):
        self.config_path = Path(config_path)
        self.custom_modules_path = Path(config_path) if custom_modules_path is not None else custom_modules_path
        self.config = yaml.safe_load(self.config_path.open())
        self.do_record = False

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
        # entites = Entities()
        entity_classes = dict()
        entities = []
        if c.DEFAULTS in self.entities:
            entities.extend(self.default_entites)
        entities.extend(x for x in self.entities if x != c.DEFAULTS)

        for entity in entities:
            try:
                folder_path = MODULE_PATH if entity not in self.default_entites else DEFAULT_PATH
                folder_path = (Path(__file__) / '..' / '..' / '..' / folder_path)
                entity_class = locate_and_import_class(entity, folder_path)
            except AttributeError:
                folder_path = self.custom_modules_path
                entity_class = locate_and_import_class(entity, folder_path)
            entity_kwargs = self.entities.get(entity, {})
            entity_symbol = entity_class.symbol if hasattr(entity_class, 'symbol') else None
            entity_classes.update({entity: {'class': entity_class, 'kwargs': entity_kwargs, 'symbol': entity_symbol}})
        return entity_classes

    def load_agents(self, size, free_tiles):
        agents = Agents(size)
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
            if c.DEFAULTS in self.agents[name]['Observations']:
                observations.extend(self.default_observations)
            observations.extend(x for x in self.agents[name]['Observations'] if x != c.DEFAULTS)
            agent = Agent(parsed_actions, observations, free_tiles.pop(), str_ident=name)
            agents.add_item(agent)
        return agents

    def load_rules(self):
        # entites = Entities()
        rules_classes = dict()
        rules = []
        if c.DEFAULTS in self.rules:
            for rule in self.default_rules:
                if rule not in rules:
                    rules.append(rule)
        rules.extend(x for x in self.rules if x != c.DEFAULTS)

        for rule in rules:
            folder_path = MODULE_PATH if rule not in self.default_rules else DEFAULT_PATH
            try:
                rule_class = locate_and_import_class(rule, folder_path)
            except AttributeError:
                rule_class = locate_and_import_class(rule, self.custom_modules_path)
            rule_kwargs = self.rules.get(rule, {})
            rules_classes.update({rule: {'class': rule_class, 'kwargs': rule_kwargs}})
        return rules_classes
