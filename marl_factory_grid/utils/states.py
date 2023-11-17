from itertools import islice
from typing import List, Tuple

import numpy as np

from marl_factory_grid.algorithms.static.utils import points_to_graph
from marl_factory_grid.environment import constants as c
from marl_factory_grid.environment.entity.entity import Entity
from marl_factory_grid.environment.rules import Rule, SpawnAgents
from marl_factory_grid.utils.results import Result, DoneResult


class StepRules:
    def __init__(self, *args):
        """
        TODO


        :return:
        """
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

    def do_all_reset(self, state):
        SpawnAgents().on_reset(state)
        for rule in self.rules:
            if rule_reset_printline := rule.on_reset(state):
                state.print(rule_reset_printline)
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
    def floortile_graph(self):
        if not self._floortile_graph:
            self.print("Generating Floorgraph....")
            self._floortile_graph = points_to_graph(self.entities.floorlist)
        return self._floortile_graph

    @property
    def moving_entites(self):
        return [y for x in self.entities for y in x if x.var_can_move]

    def __init__(self, entities, agents_conf, rules: List[Rule], lvl_shape, env_seed=69, verbose=False):
        """
        TODO


        :return:
        """
        self.lvl_shape = lvl_shape
        self.entities = entities
        self.curr_step = 0
        self.curr_actions = None
        self.agents_conf = agents_conf
        self.verbose = verbose
        self.rng = np.random.default_rng(env_seed)
        self.rules = StepRules(*rules)
        self._floortile_graph = None

    def reset(self):
        self.curr_step = 0
        self.curr_actions = None

    def __getitem__(self, item):
        return self.entities[item]

    def __iter__(self):
        return iter(e for e in self.entities.values())

    def __contains__(self, item):
        return item in self.entities

    def __repr__(self):
        return f'{self.__class__.__name__}({len(self.entities)} Entitites @ Step {self.curr_step})'

    @property
    def random_free_position(self) -> (int, int):
        """
        Returns a single **free** position (x, y), which is **free** for spawning or walking.
        No Entity at this position posses *var_is_blocking_pos* or *var_can_collide*.

        :return:    Single **free** position.
        """
        return self.get_n_random_free_positions(1)[0]

    def get_n_random_free_positions(self, n) -> list[tuple[int, int]]:
        """
        Returns a list of *n* **free** positions [(x, y), ... ], which are **free** for spawning or walking.
        No Entity at this position posses *var_is_blocking_pos* or *var_can_collide*.

        :return:    List of n **free** position.
        """
        return list(islice(self.entities.free_positions_generator, n))

    @property
    def random_position(self) -> (int, int):
        """
        Returns a single available position (x, y), ignores all entity attributes.

        :return:    Single random position.
        """
        return self.get_n_random_positions(1)[0]

    def get_n_random_positions(self, n) -> list[tuple[int, int]]:
        """
        Returns a list of *n* available positions [(x, y), ... ], ignores all entity attributes.

        :return:    List of n random positions.
        """
        return list(islice(self.entities.floorlist, n))

    def tick(self, actions) -> list[Result]:
        """
        Performs a single **Gamestate Tick**by calling the inner rule hooks in sequential order.
        - tick_pre_step_all:    Things to do before the agents do their actions. Statechange, Moving, Spawning etc...
        - agent tick:           Agents do their actions.
        - tick_step_all:        Things to do after the agents did their actions. Statechange, Moving, Spawning etc...
        - tick_post_step_all:   Things to do at the very end of each step. Counting, Reward calculations etc...

        :return:    List of *Result*-objects.
        """
        results = list()
        self.curr_step += 1

        for entity in self.entities.iter_entities():
            entity.clear_temp_state()

        # Main Agent Step
        results.extend(self.rules.tick_pre_step_all(self))

        for idx, action_int in enumerate(actions):
            agent = self[c.AGENT][idx].clear_temp_state()
            if not agent.var_is_paralyzed:
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

    def print(self, string) -> None:
        """
        When *verbose* is active, print stuff.

        :param string:      *String* to print.
        :type string:       str
        :return: Nothing
        """
        if self.verbose:
            print(string)

    def check_done(self) -> List[DoneResult]:
        """
        Iterate all **Rules** that override tehe *on_ckeck_done* hook.

        :return:    List of Results
        """
        results = list()
        for rule in self.rules:
            if on_check_done_result := rule.on_check_done(self):
                results.extend(on_check_done_result)
        return results

    def get_collision_positions(self) -> List[Tuple[(int, int)]]:
        """
        Returns a list positions [(x, y), ... ] on which collisions occur. This does not include agents,
        that were unable to move because their target direction was blocked, also a form of collision.

        :return:    List of positions.
        """
        positions = [pos for pos, entities in self.entities.pos_dict.items() if
                     len(entities) >= 2 and (len([e for e in entities if e.var_can_collide]) >= 2)
                     ]
        return positions

    def check_move_validity(self, moving_entity: Entity, target_position: (int, int)) -> bool:
        """
        Whether it is safe to move to the target positions and moving entity does not introduce a blocking attribute,
        when position is allready occupied.
        !!! Will still report true even though, there could be an enity, which var_can_collide == true !!!

        :param moving_entity: Entity
        :param target_position: pos
        :return:    Safe to move to
        """

        is_not_blocked = self.check_pos_validity(target_position)
        will_not_block_others = moving_entity.var_is_blocking_pos and self.entities.is_occupied(target_position)

        if moving_entity.pos != target_position and is_not_blocked and not will_not_block_others:
            return c.VALID
        else:
            return c.NOT_VALID

    def check_pos_validity(self, pos: (int, int)) -> bool:
        """
        Check if *pos* is a valid position to move or spawn to.

        :param pos: position to check
        :return: Wheter pos is a valid target.
        """

        if not any(e.var_is_blocking_pos for e in self.entities.pos_dict[pos]) and pos in self.entities.floorlist:
            return c.VALID
        else:
            return c.NOT_VALID
