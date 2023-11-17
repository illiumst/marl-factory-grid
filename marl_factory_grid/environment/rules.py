import abc
from random import shuffle
from typing import List, Collection

from marl_factory_grid.environment import rewards as r, constants as c
from marl_factory_grid.environment.entity.agent import Agent
from marl_factory_grid.utils import helpers as h
from marl_factory_grid.utils.results import TickResult, DoneResult


class Rule(abc.ABC):

    @property
    def name(self):
        """
        TODO


        :return:
        """
        return self.__class__.__name__

    def __init__(self):
        """
        TODO


        :return:
        """
        pass

    def __repr__(self):
        return f'{self.name}'

    def on_init(self, state, lvl_map):
        """
        TODO


        :return:
        """
        return []

    def on_reset(self, state) -> List[TickResult]:
        """
        TODO


        :return:
        """
        return []

    def tick_pre_step(self, state) -> List[TickResult]:
        """
        TODO


        :return:
        """
        return []

    def tick_step(self, state) -> List[TickResult]:
        """
        TODO


        :return:
        """
        return []

    def tick_post_step(self, state) -> List[TickResult]:
        """
        TODO


        :return:
        """
        return []

    def on_check_done(self, state) -> List[DoneResult]:
        """
        TODO


        :return:
        """
        return []


class SpawnEntity(Rule):

    @property
    def name(self):
        return f'{self.__class__.__name__}({self.collection.name})'

    def __init__(self, collection, coords_or_quantity, ignore_blocking=False):
        """
        TODO


        :return:
        """
        super().__init__()
        self.coords_or_quantity = coords_or_quantity
        self.collection = collection
        self.ignore_blocking = ignore_blocking

    def on_reset(self, state) -> [TickResult]:
        results = self.collection.trigger_spawn(state, ignore_blocking=self.ignore_blocking)
        pos_str = f' on: {[x.pos for x in self.collection]}' if self.collection.var_has_position else ''
        state.print(f'Initial {self.collection.__class__.__name__} were spawned{pos_str}')
        return results


class SpawnAgents(Rule):

    def __init__(self):
        """
        TODO


        :return:
        """
        super().__init__()
        pass

    def on_reset(self, state):
        agents = state[c.AGENT]
        for agent_name, agent_conf in state.agents_conf.items():
            empty_positions = state.entities.empty_positions
            actions = agent_conf['actions'].copy()
            observations = agent_conf['observations'].copy()
            positions = agent_conf['positions'].copy()
            other = agent_conf['other'].copy()

            if position := h.get_first(x for x in positions if x in empty_positions):
                assert state.check_pos_validity(position), 'smth went wrong....'
                agents.add_item(Agent(actions, observations, position, str_ident=agent_name, **other))
            elif positions:
                raise ValueError(f'It was not possible to spawn an Agent on the available position: '
                                 f'\n{agent_conf["positions"].copy()}')
            else:
                agents.add_item(Agent(actions, observations, empty_positions.pop(), str_ident=agent_name, **other))
        return []


class DoneAtMaxStepsReached(Rule):

    def __init__(self, max_steps: int = 500):
        """
        TODO


        :return:
        """
        super().__init__()
        self.max_steps = max_steps

    def on_check_done(self, state):
        if self.max_steps <= state.curr_step:
            return [DoneResult(validity=c.VALID, identifier=self.name)]
        return []


class AssignGlobalPositions(Rule):

    def __init__(self):
        """
        TODO


        :return:
        """
        super().__init__()

    def on_reset(self, state, lvl_map):
        from marl_factory_grid.environment.entity.util import GlobalPosition
        for agent in state[c.AGENT]:
            gp = GlobalPosition(agent, lvl_map.level_shape)
            state[c.GLOBALPOSITIONS].add_item(gp)
        return []


class WatchCollisions(Rule):

    def __init__(self, reward=r.COLLISION, done_at_collisions: bool = False, reward_at_done=r.COLLISION_DONE):
        """
        TODO


        :return:
        """
        super().__init__()
        self.reward_at_done = reward_at_done
        self.reward = reward
        self.done_at_collisions = done_at_collisions
        self.curr_done = False

    def tick_post_step(self, state) -> List[TickResult]:
        self.curr_done = False
        pos_with_collisions = state.get_collision_positions()
        results = list()
        for pos in pos_with_collisions:
            guests = [x for x in state.entities.pos_dict[pos] if x.var_can_collide]
            if len(guests) >= 2:
                for i, guest in enumerate(guests):
                    try:
                        guest.set_state(TickResult(identifier=c.COLLISION, reward=self.reward,
                                                   validity=c.NOT_VALID, entity=guest)
                                        )
                    except AttributeError:
                        pass
                    results.append(TickResult(entity=guest, identifier=c.COLLISION,
                                              reward=self.reward, validity=c.VALID))
                self.curr_done = True if self.done_at_collisions else False
        return results

    def on_check_done(self, state) -> List[DoneResult]:
        if self.done_at_collisions:
            inter_entity_collision_detected = self.curr_done
            move_failed = any(h.is_move(x.state.identifier) and not x.state.validity for x in state[c.AGENT])
            if inter_entity_collision_detected or move_failed:
                return [DoneResult(validity=c.VALID, identifier=c.COLLISION, reward=self.reward_at_done)]
        return []
