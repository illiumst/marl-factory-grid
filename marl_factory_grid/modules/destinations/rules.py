import ast
from random import shuffle
from typing import List, Dict, Tuple

from marl_factory_grid.environment.rules import Rule
from marl_factory_grid.utils import helpers as h
from marl_factory_grid.utils.results import TickResult, DoneResult
from marl_factory_grid.environment import constants as c

from marl_factory_grid.modules.destinations import constants as d
from marl_factory_grid.modules.destinations.entitites import Destination
from marl_factory_grid.utils.states import Gamestate

ANY = 'any'
ALL = 'all'
SIMULTANEOUS = 'simultanious'
CONDITIONS = [ALL, ANY, SIMULTANEOUS]


class DestinationReachReward(Rule):

    def __init__(self, dest_reach_reward=d.REWARD_DEST_REACHED):
        """
        This rule introduces the basic functionality, so that targets (Destinations) can be reached and marked as such.
        Additionally, rewards are reported.

        :type dest_reach_reward: float
        :param dest_reach_reward: Specifies the reward, agents get at destination reach.

        """
        super(DestinationReachReward, self).__init__()
        self.reward = dest_reach_reward

    def tick_step(self, state) -> List[TickResult]:
        results = []
        reached = False
        for dest in state[d.DESTINATION]:
            if dest.has_just_been_reached(state) and not dest.was_reached():
                # Dest has just been reached, some agent needs to stand here
                for agent in state[c.AGENT].by_pos(dest.pos):
                    if dest.bound_entity:
                        if dest.bound_entity == agent:
                            reached = True
                        else:
                            pass
                    else:
                        reached = True
            else:
                pass
            if reached:
                state.print(f'{dest.name} is reached now, mark as reached...')
                dest.mark_as_reached()
                results.append(TickResult(self.name, validity=c.VALID, reward=self.reward, entity=agent))
        return results


class DoneAtDestinationReach(DestinationReachReward):

    def __init__(self, condition='any', reward_at_done=d.REWARD_DEST_DONE, **kwargs):
        """
        This rule triggers and sets the done flag if ALL Destinations have been reached.

        :type reward_at_done: float
        :param reward_at_done: Specifies the reward, agent get, when all destinations are reached.
        :type dest_reach_reward: float
        :param dest_reach_reward: Specify the reward, agents get when reaching a single destination.
        """
        super().__init__(**kwargs)
        self.condition = condition
        self.reward = reward_at_done
        assert condition in CONDITIONS

    def on_check_done(self, state) -> List[DoneResult]:
        if self.condition == ANY:
            if any(x.was_reached() for x in state[d.DESTINATION]):
                return [DoneResult(self.name, validity=c.VALID, reward=self.reward)]
        elif self.condition == ALL:
            if all(x.was_reached() for x in state[d.DESTINATION]):
                return [DoneResult(self.name, validity=c.VALID, reward=self.reward)]
        elif self.condition == SIMULTANEOUS:
            if all(x.was_reached() for x in state[d.DESTINATION]):
                return [DoneResult(self.name, validity=c.VALID, reward=self.reward)]
            else:
                for dest in state[d.DESTINATION]:
                    if dest.was_reached():
                        for agent in state[c.AGENT].by_pos(dest.pos):
                            if dest.bound_entity:
                                if dest.bound_entity == agent:
                                    pass
                                else:
                                    dest.unmark_as_reached()
                                    return [DoneResult(f'{dest}_unmarked_as_reached',
                                                       validity=c.NOT_VALID, entity=dest)]
                            else:
                                pass
        else:
            raise ValueError('Check spelling of Parameter "condition".')


class SpawnDestinationsPerAgent(Rule):
    def __init__(self, coords_or_quantity: Dict[str, List[Tuple[int, int] | int]]):
        """
        Special rule, that spawn destinations, that are bound to a single agent a fixed set of positions.
        Useful for introducing specialists, etc. ..

        !!! This rule does not introduce any reward or done condition.

        :param coords_or_quantity: Please provide a dictionary with agent names as keys; and a list of possible
                                     destination coords as value. Example: {Wolfgang: [(0, 0), (1, 1), ...]}
        """
        super(Rule, self).__init__()
        self.per_agent_positions = dict()
        for agent_name, value in coords_or_quantity.items():
            if isinstance(value, int):
                per_agent_d = {agent_name: value}
            else:
                per_agent_d = {agent_name: [ast.literal_eval(x) for x in value]}
            self.per_agent_positions.update(**per_agent_d)

    def on_reset(self, state: Gamestate):
        for (agent_name, coords_or_quantity) in self.per_agent_positions.items():
            agent = h.get_first(state[c.AGENT], lambda x: agent_name in x.name)
            assert agent
            if isinstance(coords_or_quantity, int):
                position_list = state.entities.floorlist
                pos_left_counter = coords_or_quantity
            else:
                position_list = coords_or_quantity.copy()
                pos_left_counter = 1  # Find a better way to resolve this.
            shuffle(position_list)
            while pos_left_counter:
                try:
                    pos = position_list.pop()
                except IndexError:
                    print(f"Could not spawn Destinations at: {self.per_agent_positions[agent_name]}")
                    print(f'Check your agent placement: {state[c.AGENT]} ... Exit ...')
                    exit(-9999)
                if (not pos == agent.pos) and (not state[d.DESTINATION].by_pos(pos)):
                    destination = Destination(pos, bind_to=agent)
                    pos_left_counter -= 1
                    break
                else:
                    continue
            state[d.DESTINATION].add_item(destination)
        pass
