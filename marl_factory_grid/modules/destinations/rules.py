import ast
from random import shuffle
from typing import List, Dict, Tuple

from marl_factory_grid.environment.rules import Rule
from marl_factory_grid.utils import helpers as h
from marl_factory_grid.utils.results import TickResult, DoneResult
from marl_factory_grid.environment import constants as c

from marl_factory_grid.modules.destinations import constants as d
from marl_factory_grid.modules.destinations.entitites import Destination


class DestinationReachReward(Rule):

    def __init__(self, dest_reach_reward=d.REWARD_DEST_REACHED):
        """
        This rule introduces the basic functionality, so that targts (Destinations) can be reached and marked as such.
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


class DoneAtDestinationReachAll(DestinationReachReward):

    def __init__(self, reward_at_done=d.REWARD_DEST_DONE, **kwargs):
        """
        This rule triggers and sets the done flag if ALL Destinations have been reached.

        :type reward_at_done: float
        :param reward_at_done: Specifies the reward, agent get, whenn all destinations are reached.
        :type dest_reach_reward: float
        :param dest_reach_reward: Specify the reward, agents get when reaching a single destination.
        """
        super(DoneAtDestinationReachAll, self).__init__(**kwargs)
        self.reward = reward_at_done

    def on_check_done(self, state) -> List[DoneResult]:
        if all(x.was_reached() for x in state[d.DESTINATION]):
            return [DoneResult(self.name, validity=c.VALID, reward=self.reward)]
        return [DoneResult(self.name, validity=c.NOT_VALID)]


class DoneAtDestinationReachAny(DestinationReachReward):

    def __init__(self, reward_at_done=d.REWARD_DEST_DONE, **kwargs):
        f"""
        This rule triggers and sets the done flag if ANY Destinations has been reached.
        !!! IMPORTANT: 'reward_at_done' is shared between the agents; 'dest_reach_reward' is bound to a specific one.
                
        :type reward_at_done: float
        :param reward_at_done: Specifies the reward, all agent get, when any destinations has been reached. 
                                Default {d.REWARD_DEST_DONE}
        :type dest_reach_reward: float
        :param dest_reach_reward: Specify a single agents reward forreaching a single destination. 
                                   Default {d.REWARD_DEST_REACHED}
        """
        super(DoneAtDestinationReachAny, self).__init__(**kwargs)
        self.reward = reward_at_done

    def on_check_done(self, state) -> List[DoneResult]:
        if any(x.was_reached() for x in state[d.DESTINATION]):
            return [DoneResult(self.name, validity=c.VALID, reward=d.REWARD_DEST_REACHED)]
        return []


class SpawnDestinationsPerAgent(Rule):
    def __init__(self, coords_or_quantity: Dict[str, List[Tuple[int, int]]]):
        """
        Special rule, that spawn distinations, that are bound to a single agent a fixed set of positions.
        Usefull for introducing specialists, etc. ..

        !!! This rule does not introduce any reward or done condition.

        :type coords_or_quantity:  Dict[str, List[Tuple[int, int]]
        :param coords_or_quantity: Please provide a dictionary with agent names as keys; and a list of possible
                                     destiantion coords as value. Example: {Wolfgang: [(0, 0), (1, 1), ...]}
        """
        super(Rule, self).__init__()
        self.per_agent_positions = {key: [ast.literal_eval(x) for x in val] for key, val in coords_or_quantity.items()}

    def on_init(self, state, lvl_map):
        for (agent_name, position_list) in self.per_agent_positions.items():
            agent = h.get_first(state[c.AGENT], lambda x: agent_name in x.name)
            assert agent
            position_list = position_list.copy()
            shuffle(position_list)
            while True:
                try:
                    pos = position_list.pop()
                except IndexError:
                    print(f"Could not spawn Destinations at: {self.per_agent_positions[agent_name]}")
                    print(f'Check your agent placement: {state[c.AGENT]} ... Exit ...')
                    exit(9999)
                if (not pos == agent.pos) and (not state[d.DESTINATION].by_pos(pos)):
                    destination = Destination(pos, bind_to=agent)
                    break
                else:
                    continue
            state[d.DESTINATION].add_item(destination)
        pass
