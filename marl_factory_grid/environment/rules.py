import abc
import random
from random import shuffle
from typing import List, Collection

import numpy as np

import marl_factory_grid
from marl_factory_grid.environment import rewards as r, constants as c
from marl_factory_grid.environment.entity.agent import Agent
from marl_factory_grid.utils import helpers as h
from marl_factory_grid.utils.results import TickResult, DoneResult


class Rule(abc.ABC):

    @property
    def name(self):
        """
       Get the name of the rule.

       :return: The name of the rule.
       :rtype: str
       """
        return self.__class__.__name__

    def __init__(self):
        """
        Abstract base class representing a rule in the environment.

        This class provides a framework for defining rules that govern the behavior of the environment. Rules can be
        implemented by inheriting from this class and overriding specific methods.

        """
        pass

    def __repr__(self) -> str:
        """
        Return a string representation of the rule.

        :return: A string representation of the rule.
        :rtype: str
        """
        return f'{self.name}'

    def on_init(self, state, lvl_map):
        """
        Initialize the rule when the environment is created.

        This method is called during the initialization of the environment. It allows the rule to perform any setup or
        initialization required.

        :param state: The current game state.
        :type state: marl_factory_grid.utils.states.GameState
        :param lvl_map: The map of the level.
        :type lvl_map: marl_factory_grid.environment.level.LevelMap
        :return: List of TickResults generated during initialization.
        :rtype: List[TickResult]
        """
        return []

    def on_reset_post_spawn(self, state) -> List[TickResult]:
        """
        Execute actions after entities are spawned during a reset.

        This method is called after entities are spawned during a reset. It allows the rule to perform any actions
        required at this stage.

        :param state: The current game state.
        :type state: marl_factory_grid.utils.states.GameState
        :return: List of TickResults generated after entity spawning.
        :rtype: List[TickResult]
        """
        return []

    def on_reset(self, state) -> List[TickResult]:
        """
        Execute actions during a reset.

        This method is called during a reset. It allows the rule to perform any actions required at this stage.

        :param state: The current game state.
        :type state: marl_factory_grid.utils.states.GameState
        :return: List of TickResults generated during a reset.
        :rtype: List[TickResult]
        """
        return []

    def tick_pre_step(self, state) -> List[TickResult]:
        """
        Execute actions before the main step of the environment.

        This method is called before the main step of the environment. It allows the rule to perform any actions
        required before the main step.

        :param state: The current game state.
        :type state: marl_factory_grid.utils.states.GameState
        :return: List of TickResults generated before the main step.
        :rtype: List[TickResult]
        """
        return []

    def tick_step(self, state) -> List[TickResult]:
        """
        Execute actions during the main step of the environment.

        This method is called during the main step of the environment. It allows the rule to perform any actions
        required during the main step.

        :param state: The current game state.
        :type state: marl_factory_grid.utils.states.GameState
        :return: List of TickResults generated during the main step.
        :rtype: List[TickResult]
        """
        return []

    def tick_post_step(self, state) -> List[TickResult]:
        """
        Execute actions after the main step of the environment.

        This method is called after the main step of the environment. It allows the rule to perform any actions
        required after the main step.

        :param state: The current game state.
        :type state: marl_factory_grid.utils.states.GameState
        :return: List of TickResults generated after the main step.
        :rtype: List[TickResult]
        """
        return []

    def on_check_done(self, state) -> List[DoneResult]:
        """
        Check conditions for the termination of the environment.

        This method is called to check conditions for the termination of the environment. It allows the rule to
        specify conditions under which the environment should be considered done.

        :param state: The current game state.
        :type state: marl_factory_grid.utils.states.GameState
        :return: List of DoneResults indicating whether the environment is done.
        :rtype: List[DoneResult]
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


def _get_position(spawn_rule, positions, empty_positions, positions_pointer):
    """
    Internal usage, selects positions based on rule.
    """
    if spawn_rule and spawn_rule == "random":
        position = random.choice(([x for x in positions if x in empty_positions]))
    elif spawn_rule and spawn_rule == "order":
        position = ([x for x in positions if x in empty_positions])[positions_pointer]
    else:
        position = h.get_first([x for x in positions if x in empty_positions])
    return position


class SpawnAgents(Rule):

    def __init__(self):
        """
        Finds suitable spawn positions according to the given spawn rule, creates agents with these positions and adds
        them to state.agents.
        """
        super().__init__()
        pass

    def on_reset(self, state):
        spawn_rule = None
        for rule in state.rules.rules:
            if isinstance(rule, AgentSpawnRule):
                spawn_rule = rule.spawn_rule
                break

        if not hasattr(state, 'agent_spawn_positions'):
            state.agent_spawn_positions = []
        else:
            state.agent_spawn_positions.clear()

        agents = state[c.AGENT]
        for agent_name, agent_conf in state.agents_conf.items():
            empty_positions = state.entities.empty_positions
            actions = agent_conf['actions'].copy()
            observations = agent_conf['observations'].copy()
            positions = agent_conf['positions'].copy()
            other = agent_conf['other'].copy()
            positions_pointer = agent_conf['pos_pointer']

            if position := _get_position(spawn_rule, positions, empty_positions, positions_pointer):
                assert state.check_pos_validity(position), 'smth went wrong....'
                agents.add_item(Agent(actions, observations, position, str_ident=agent_name, **other))
                state.agent_spawn_positions.append(position)
            elif positions:
                raise ValueError(f'It was not possible to spawn an Agent on the available position: '
                                 f'\n{agent_conf["positions"].copy()}')
            else:
                chosen_position = empty_positions.pop()
                agents.add_item(Agent(actions, observations, chosen_position, str_ident=agent_name, **other))
                state.agent_spawn_positions.append(chosen_position)
        return []


class AgentSpawnRule(Rule):
    def __init__(self, spawn_rule):
        self.spawn_rule = spawn_rule
        super().__init__()


class DoneAtMaxStepsReached(Rule):

    def __init__(self, max_steps: int = 500):
        """
       A rule that terminates the environment when a specified maximum number of steps is reached.

       :param max_steps: The maximum number of steps before the environment is considered done.
       :type max_steps: int
       """
        super().__init__()
        self.max_steps = max_steps

    def on_check_done(self, state):
        """
        Check if the maximum number of steps is reached, and if so, mark the environment as done.

       :param state: The current game state.
       :type state: marl_factory_grid.utils.states.GameState
       :return: List of DoneResults indicating whether the environment is done.
       :rtype: List[DoneResult]
       """
        if self.max_steps <= state.curr_step:
            return [DoneResult(validity=c.VALID, identifier=self.name)]
        return []


class AssignGlobalPositions(Rule):

    def __init__(self):
        """
        A rule that assigns global positions to agents when the environment is reset.

        :return: None
        """
        super().__init__()
        self.level_shape = None

    def on_init(self, state, lvl_map):
        self.level_shape = lvl_map.level_shape

    def on_reset(self, state):
        """
       Assign global positions to agents when the environment is reset.

       :param state: The current game state.
       :type state: marl_factory_grid.utils.states.GameState
       :param lvl_map: The map of the current level.
       :type lvl_map: marl_factory_grid.levels.level.LevelMap
       :return: An empty list, as no additional results are generated by this rule during the reset.
       :rtype: List[TickResult]
       """
        from marl_factory_grid.environment.entity.util import GlobalPosition
        for agent in state[c.AGENT]:
            gp = GlobalPosition(agent, self.level_shape)
            state[c.GLOBALPOSITIONS].add_item(gp)
        return []


class WatchCollisions(Rule):

    def __init__(self, reward=r.COLLISION, done_at_collisions: bool = False, reward_at_done=r.COLLISION_DONE):
        """
        A rule that monitors collisions between entities in the environment.

        :param reward: The reward assigned for each collision.
        :type reward: float
        :param done_at_collisions: If True, marks the environment as done when collisions occur.
        :type done_at_collisions: bool
        :param reward_at_done: The reward assigned when the environment is marked as done due to collisions.
        :type reward_at_done: float
        :return: None
        """
        super().__init__()
        self.reward_at_done = reward_at_done
        self.reward = reward
        self.done_at_collisions = done_at_collisions
        self.curr_done = False

    def tick_post_step(self, state) -> List[TickResult]:
        """
        Monitors collisions between entities after each step in the environment.

        :param state: The current game state.
        :type state: marl_factory_grid.utils.states.GameState
        :return: A list of TickResult objects representing collisions and their associated rewards.
        :rtype: List[TickResult]
        """
        self.curr_done = False
        results = list()
        for agent in state[c.AGENT]:
            a_s = agent.state
            if h.is_move(a_s.identifier) and a_s.action_introduced_collision:
                results.append(TickResult(entity=agent, identifier=c.COLLISION,
                                          reward=self.reward, validity=c.VALID))

        for pos in state.get_collision_positions():
            guests = [x for x in state.entities.pos_dict[pos] if x.var_can_collide]
            if len(guests) >= 2:
                for i, guest in enumerate(guests):
                    try:
                        guest.set_state(TickResult(identifier=c.COLLISION, reward=self.reward,
                                                   validity=c.NOT_VALID, entity=guest)
                                        )
                    except AttributeError:
                        pass
                    if not any([x.entity == guest for x in results]):
                        results.append(TickResult(entity=guest, identifier=c.COLLISION,
                                                  reward=self.reward, validity=c.VALID))
                self.curr_done = True if self.done_at_collisions else False
        return results

    def on_check_done(self, state) -> List[DoneResult]:
        """
        Checks if the environment should be marked as done based on collision conditions.

        :param state: The current game state.
        :type state: marl_factory_grid.utils.states.GameState
        :return: A list of DoneResult objects representing the conditions for marking the environment as done.
        :rtype: List[DoneResult]
        """
        if self.done_at_collisions:
            inter_entity_collision_detected = self.curr_done
            collision_in_step = any(h.is_move(x.state.identifier) and x.state.action_introduced_collision
                                    for x in state[c.AGENT]
                                    )
            if inter_entity_collision_detected or collision_in_step:
                return [DoneResult(validity=c.VALID, identifier=c.COLLISION, reward=self.reward_at_done)]
        return []


class DoRandomInitialSteps(Rule):
    def __init__(self, random_steps: 10):
        """
        Special rule which spawns destinations, that are bound to a single agent a fixed set of positions.
        Useful for introducing specialists, etc. ..

        !!! This rule does not introduce any reward or done condition.

        :param random_steps:  Number of random steps agents perform in an environment.
                                Useful in the `N-Puzzle` configuration.
        """
        super().__init__()
        self.random_steps = random_steps

    def on_reset_post_spawn(self, state):
        state.print("Random Initial Steps initiated....")
        for _ in range(self.random_steps):
            # Find free positions
            free_pos = state.random_free_position
            neighbor_positions = state.entities.neighboring_4_positions(free_pos)
            random.shuffle(neighbor_positions)
            chosen_agent = h.get_first(state[c.AGENT].by_pos(neighbor_positions.pop()))
            assert isinstance(chosen_agent, Agent)
            valid = chosen_agent.move(free_pos, state)
            valid_str = " not" if not valid else ""
            state.print(f"Move {chosen_agent.name} from {chosen_agent.last_pos} "
                        f"to {chosen_agent.pos} was{valid_str} valid.")
        pass
