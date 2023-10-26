from typing import List, Dict, Tuple

import numpy as np

from marl_factory_grid.environment import constants as c
from marl_factory_grid.environment.rules import Rule
from marl_factory_grid.utils.results import Result


class StepRules:
    def __init__(self, *args):
        if args:
            self.rules = list(args)
        else:
            self.rules = list()

    def __repr__(self):
        return f'Rules{[x.name for x in self]}'

    def __iter__(self):
        return iter(self.rules)

    def append(self, item):
        assert isinstance(item, Rule)
        self.rules.append(item)
        return True

    def do_all_init(self, state, lvl_map):
        for rule in self.rules:
            if rule_init_printline := rule.on_init(state, lvl_map):
                state.print(rule_init_printline)
        return c.VALID

    def tick_step_all(self, state):
        results = list()
        for rule in self.rules:
            if tick_step_result := rule.tick_step(state):
                results.extend(tick_step_result)
        return results

    def tick_pre_step_all(self, state):
        results = list()
        for rule in self.rules:
            if tick_pre_step_result := rule.tick_pre_step(state):
                results.extend(tick_pre_step_result)
        return results

    def tick_post_step_all(self, state):
        results = list()
        for rule in self.rules:
            if tick_post_step_result := rule.tick_post_step(state):
                results.extend(tick_post_step_result)
        return results


class Gamestate(object):

    @property
    def moving_entites(self):
        return [y for x in self.entities for y in x if x.var_can_move]  # wird das aus dem String gelesen?

    def __init__(self, entities, agents_conf, rules: Dict[str, dict], env_seed=69, verbose=False):
        self.entities = entities
        self.curr_step = 0
        self.curr_actions = None
        self.agents_conf = agents_conf
        self.verbose = verbose
        self.rng = np.random.default_rng(env_seed)
        self.rules = StepRules(*(v['class'](**v['kwargs']) for v in rules.values()))

    def __getitem__(self, item):
        return self.entities[item]

    def __iter__(self):
        return iter(e for e in self.entities.values())

    def __contains__(self, item):
        return item in self.entities

    def __repr__(self):
        return f'{self.__class__.__name__}({len(self.entities)} Entitites @ Step {self.curr_step})'

    def tick(self, actions) -> List[Result]:
        results = list()
        self.curr_step += 1

        # Main Agent Step
        results.extend(self.rules.tick_pre_step_all(self))

        for idx, action_int in enumerate(actions):
            if not agent.var_is_paralyzed:
                agent = self[c.AGENT][idx].clear_temp_state()
                action = agent.actions[action_int]
                action_result = action.do(agent, self)
                results.append(action_result)
                agent.set_state(action_result)
            else:
                self.print(f"{agent.name} is paralied because of: {agent.paralyze_reasons}")
                continue

        results.extend(self.rules.tick_step_all(self))
        results.extend(self.rules.tick_post_step_all(self))

        return results

    def print(self, string):
        if self.verbose:
            print(string)

    def check_done(self):
        results = list()
        for rule in self.rules:
            if on_check_done_result := rule.on_check_done(self):
                results.extend(on_check_done_result)
        return results


    def get_all_pos_with_collisions(self) -> List[Tuple[(int, int)]]:
        positions = [pos for pos, entity_list_for_position in self.entities.pos_dict.items()
                     if any([e.var_can_collide for e in entity_list_for_position])]
        return positions

    def check_move_validity(self, moving_entity, position):
        #         if (guest.name not in self._guests and not self.is_blocked)
        #         and not (guest.var_is_blocking_pos and self.is_occupied()):
        if moving_entity.pos != position and not any(
                entity.var_is_blocking_pos for entity in self.entities.pos_dict[position]) and not (
                moving_entity.var_is_blocking_pos and self.entities.is_occupied(position)):
            return True
        else:
            return False

    def check_pos_validity(self, position):
        if not any(entity.var_is_blocking_pos for entity in self.entities.pos_dict[position]):
            return True
        else:
            return False

