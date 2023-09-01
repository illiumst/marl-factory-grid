from typing import List, Union
from marl_factory_grid.environment.rules import Rule
from marl_factory_grid.utils.results import TickResult, DoneResult
from marl_factory_grid.environment import constants as c

from marl_factory_grid.modules.destinations import constants as d, rewards as r
from marl_factory_grid.modules.destinations.entitites import Destination


class DestinationReach(Rule):

    def __init__(self, n_dests: int = 1, tiles: Union[List, None] = None):
        super(DestinationReach, self).__init__()
        self.n_dests = n_dests or len(tiles)
        self._tiles = tiles

    def tick_step(self, state) -> List[TickResult]:

        for dest in list(state[d.DESTINATION].values()):
            if dest.is_considered_reached:
                dest.change_parent_collection(state[d.DEST_REACHED])
                state.print(f'{dest.name} is reached now, removing...')
            else:
                for agent_name in dest.currently_dwelling_names:
                    agent = state[c.AGENT][agent_name]
                    if agent.pos == dest.pos:
                        state.print(f'{agent.name} is still waiting.')
                        pass
                    else:
                        dest.leave(agent)
                        state.print(f'{agent.name} left the destination early.')
        return [TickResult(self.name, validity=c.VALID, reward=0, entity=None)]

    def tick_post_step(self, state) -> List[TickResult]:
        results = list()
        for reached_dest in state[d.DEST_REACHED]:
            for guest in reached_dest.tile.guests:
                if guest in state[c.AGENT]:
                    state.print(f'{guest.name} just reached destination at {guest.pos}')
                    state[d.DEST_REACHED].delete_env_object(reached_dest)
                    results.append(TickResult(self.name, validity=c.VALID, reward=r.DEST_REACHED, entity=guest))
        return results


class DestinationDone(Rule):

    def __init__(self):
        super(DestinationDone, self).__init__()

    def on_check_done(self, state) -> List[DoneResult]:
        if not len(state[d.DESTINATION]):
            return [DoneResult(self.name, validity=c.VALID, reward=r.DEST_REACHED)]
        return []


class DoneOnReach(Rule):

    def __init__(self):
        super(DoneOnReach, self).__init__()

    def on_check_done(self, state) -> List[DoneResult]:
        dests = [x.pos for x in state[d.DESTINATION]]
        agents = [x.pos for x in state[c.AGENT]]

        if any([x in dests for x in agents]):
            return [DoneResult(self.name, validity=c.VALID, reward=r.DEST_REACHED)]
        return [DoneResult(self.name, validity=c.NOT_VALID, reward=0)]


class DestinationSpawn(Rule):

    def __init__(self, spawn_frequency: int = 5, n_dests: int = 1,
                 spawn_mode: str = d.MODE_GROUPED):
        super(DestinationSpawn, self).__init__()
        self.spawn_frequency = spawn_frequency
        self.n_dests = n_dests
        self.spawn_mode = spawn_mode

    def on_init(self, state, lvl_map):
        # noinspection PyAttributeOutsideInit
        self._dest_spawn_timer = self.spawn_frequency
        self.trigger_destination_spawn(self.n_dests, state)
        pass

    def tick_pre_step(self, state) -> List[TickResult]:
        pass

    def tick_step(self, state) -> List[TickResult]:
        if n_dest_spawn := max(0, self.n_dests - len(state[d.DESTINATION])):
            if self.spawn_mode == d.MODE_GROUPED and n_dest_spawn == self.n_dests:
                validity = state.rules['DestinationReach'].trigger_destination_spawn(n_dest_spawn, state)
                return [TickResult(self.name, validity=validity, entity=None, value=n_dest_spawn)]

    @staticmethod
    def trigger_destination_spawn(n_dests, state, tiles=None):
        tiles = tiles or state[c.FLOOR].empty_tiles[:n_dests]
        if destinations := [Destination(tile) for tile in tiles]:
            state[d.DESTINATION].add_items(destinations)
            state.print(f'{n_dests} new destinations have been spawned')
            return c.VALID
        else:
            state.print('No Destiantions are spawning, limit is reached.')
            return c.NOT_VALID
