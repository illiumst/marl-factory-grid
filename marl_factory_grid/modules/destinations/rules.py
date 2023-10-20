import ast
from random import shuffle
from typing import List, Dict, Tuple
from marl_factory_grid.environment.rules import Rule
from marl_factory_grid.utils.results import TickResult, DoneResult
from marl_factory_grid.environment import constants as c

from marl_factory_grid.modules.destinations import constants as d, rewards as r
from marl_factory_grid.modules.destinations.entitites import Destination


class DestinationReachAll(Rule):

    def __init__(self):
        super(DestinationReachAll, self).__init__()

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
                results.append(TickResult(self.name, validity=c.VALID, reward=r.DEST_REACHED, entity=agent))
        return results


    def on_check_done(self, state) -> List[DoneResult]:
        if all(x.was_reached() for x in state[d.DESTINATION]):
            return [DoneResult(self.name, validity=c.VALID, reward=r.DEST_REACHED)]
        return [DoneResult(self.name, validity=c.NOT_VALID, reward=0)]


class DestinationReachAny(DestinationReachAll):

    def __init__(self):
        super(DestinationReachAny, self).__init__()

    def on_check_done(self, state) -> List[DoneResult]:
        if any(x.was_reached() for x in state[d.DESTINATION]):
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
        state[d.DESTINATION].trigger_destination_spawn(self.n_dests, state)
        pass

    def tick_pre_step(self, state) -> List[TickResult]:
        pass

    def tick_step(self, state) -> List[TickResult]:
        if n_dest_spawn := max(0, self.n_dests - len(state[d.DESTINATION])):
            if self.spawn_mode == d.MODE_GROUPED and n_dest_spawn == self.n_dests:
                validity = state[d.DESTINATION].trigger_destination_spawn(n_dest_spawn, state)
                return [TickResult(self.name, validity=validity, entity=None, value=n_dest_spawn)]
            elif self.spawn_mode == d.MODE_SINGLE and n_dest_spawn:
                validity = state[d.DESTINATION].trigger_destination_spawn(n_dest_spawn, state)
                return [TickResult(self.name, validity=validity, entity=None, value=n_dest_spawn)]
            else:
                pass


class FixedDestinationSpawn(Rule):
    def __init__(self, per_agent_positions: Dict[str, List[Tuple[int, int]]]):
        super(Rule, self).__init__()
        self.per_agent_positions = {key: [ast.literal_eval(x) for x in val] for key, val in per_agent_positions.items()}

    def on_init(self, state, lvl_map):
        for (agent_name, position_list) in self.per_agent_positions.items():
            agent = next(x for x in state[c.AGENT] if agent_name in x.name)  # Fixme: Ugly AF
            position_list = position_list.copy()
            shuffle(position_list)
            while True:
                try:
                    pos = position_list.pop()
                except IndexError:
                    print(f"Could not spawn Destinations at: {self.per_agent_positions[agent_name]}")
                    print(f'Check your agent palcement: {state[c.AGENT]} ... Exit ...')
                    exit(9999)
                if (not pos == agent.pos) and (not state[d.DESTINATION].by_pos(pos)):
                    destination = Destination(pos, bind_to=agent)
                    break
                else:
                    continue
            state[d.DESTINATION].add_item(destination)
        pass
