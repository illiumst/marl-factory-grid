from collections import defaultdict

from marl_factory_grid.environment.entity.agent import Agent
from marl_factory_grid.environment.entity.entity import Entity
from marl_factory_grid.environment import constants as c
from marl_factory_grid.environment.entity.mixin import BoundEntityMixin
from marl_factory_grid.utils.render import RenderEntity
from marl_factory_grid.modules.destinations import constants as d


class Destination(Entity):

    var_can_move = False
    var_can_collide = False
    var_has_position = True
    var_is_blocking_pos = False
    var_is_blocking_light = False

    @property
    def any_agent_has_dwelled(self):
        return bool(len(self._per_agent_times))

    @property
    def currently_dwelling_names(self):
        return list(self._per_agent_times.keys())

    @property
    def encoding(self):
        return d.DEST_SYMBOL

    def __init__(self, *args, dwell_time: int = 0, **kwargs):
        super(Destination, self).__init__(*args, **kwargs)
        self.dwell_time = dwell_time
        self._per_agent_times = defaultdict(lambda: dwell_time)

    def do_wait_action(self, agent: Agent):
        self._per_agent_times[agent.name] -= 1
        return c.VALID

    def leave(self, agent: Agent):
        del self._per_agent_times[agent.name]

    @property
    def is_considered_reached(self):
        agent_at_position = any(c.AGENT.lower() in x.name.lower() for x in self.tile.guests_that_can_collide)
        return (agent_at_position and not self.dwell_time) or any(x == 0 for x in self._per_agent_times.values())

    def agent_is_dwelling(self, agent: Agent):
        return self._per_agent_times[agent.name] < self.dwell_time

    def summarize_state(self) -> dict:
        state_summary = super().summarize_state()
        state_summary.update(per_agent_times=[
            dict(belongs_to=key, time=val) for key, val in self._per_agent_times.items()], dwell_time=self.dwell_time)
        return state_summary

    def render(self):
        return RenderEntity(d.DESTINATION, self.pos)


class BoundDestination(BoundEntityMixin, Destination):

    @property
    def encoding(self):
        return d.DEST_SYMBOL

    def __init__(self, entity, *args, **kwargs):
        self.bind_to(entity)
        super().__init__(*args, **kwargs)


    @property
    def is_considered_reached(self):
        agent_at_position = any(self.bound_entity == x for x in self.tile.guests_that_can_collide)
        return (agent_at_position and not self.dwell_time) \
            or any(x == 0 for x in self._per_agent_times[self.bound_entity.name])
