import ast
from random import shuffle
from typing import List, Union, Dict, Tuple
from marl_factory_grid.environment.rules import Rule
from marl_factory_grid.utils.results import TickResult, DoneResult
from marl_factory_grid.environment import constants as c

from marl_factory_grid.modules.destinations import constants as d, rewards as r
from marl_factory_grid.modules.destinations.entitites import Destination, BoundDestination


class DestinationReachAll(Rule):

    def __init__(self):
        super(DestinationReachAll, self).__init__()

    def tick_step(self, state) -> List[TickResult]:
        results = []
        for dest in list(state[next(key for key in state.entities.names if d.DESTINATION in key)]):
            if dest.is_considered_reached:
                agent = state[c.AGENT].by_pos(dest.pos)
                results.append(TickResult(self.name, validity=c.VALID, reward=r.DEST_REACHED, entity=agent))
                state.print(f'{dest.name} is reached now, removing...')
                assert dest.destroy(), f'{dest.name} could not be destroyed. Critical Error.'
            else:
               pass
        return [TickResult(self.name, validity=c.VALID, reward=0, entity=None)]

    def tick_post_step(self, state) -> List[TickResult]:
        return []

    def on_check_done(self, state) -> List[DoneResult]:
        if not len(state[next(key for key in state.entities.names if d.DESTINATION in key)]):
            return [DoneResult(self.name, validity=c.VALID, reward=r.DEST_REACHED)]
        return [DoneResult(self.name, validity=c.NOT_VALID, reward=0)]


class DestinationReachAny(DestinationReachAll):

    def __init__(self):
        super(DestinationReachAny, self).__init__()

    def on_check_done(self, state) -> List[DoneResult]:
        if not len(state[next(key for key in state.entities.names if d.DESTINATION in key)]):
            return [DoneResult(self.name, validity=c.VALID, reward=r.DEST_REACHED)]
        return []


class DestinationSpawn(Rule):

    def __init__(self, n_dests: int = 1,
                 spawn_mode: str = d.MODE_GROUPED):
        super(DestinationSpawn, self).__init__()
        self.n_dests = n_dests
        self.spawn_mode = spawn_mode

    def on_init(self, state, lvl_map):
        # noinspection PyAttributeOutsideInit
        self.trigger_destination_spawn(self.n_dests, state)
        pass

    def tick_pre_step(self, state) -> List[TickResult]:
        pass

    def tick_step(self, state) -> List[TickResult]:
        if n_dest_spawn := max(0, self.n_dests - len(state[d.DESTINATION])):
            if self.spawn_mode == d.MODE_GROUPED and n_dest_spawn == self.n_dests:
                validity = self.trigger_destination_spawn(n_dest_spawn, state)
                return [TickResult(self.name, validity=validity, entity=None, value=n_dest_spawn)]
            elif self.spawn_mode == d.MODE_SINGLE and n_dest_spawn:
                validity = self.trigger_destination_spawn(n_dest_spawn, state)
                return [TickResult(self.name, validity=validity, entity=None, value=n_dest_spawn)]
            else:
                pass

    def trigger_destination_spawn(self, n_dests, state):
        empty_positions = state[c.FLOORS].empty_tiles[:n_dests]
        if destinations := [Destination(pos) for pos in empty_positions]:
            state[d.DESTINATION].add_items(destinations)
            state.print(f'{n_dests} new destinations have been spawned')
            return c.VALID
        else:
            state.print('No Destiantions are spawning, limit is reached.')
            return c.NOT_VALID


class FixedDestinationSpawn(Rule):
    def __init__(self, per_agent_positions: Dict[str, List[Tuple[int, int]]]):
        super(Rule, self).__init__()
        self.per_agent_positions = {key: [ast.literal_eval(x) for x in val] for key, val in per_agent_positions.items()}

    def on_init(self, state, lvl_map):
        for (agent_name, position_list) in self.per_agent_positions.items():
            agent = next(x for x in state[c.AGENT] if agent_name in x.name)  # Fixme: Ugly AF
            shuffle(position_list)
            while True:
                pos = position_list.pop()
                if pos != agent.pos and not state[d.BOUNDDESTINATION].by_pos(pos):
                    destination = BoundDestination(agent, state[c.FLOORS].by_pos(pos))
                    break
                else:
                    continue
            state[d.BOUNDDESTINATION].add_item(destination)
        pass
