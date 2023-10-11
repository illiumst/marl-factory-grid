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
    def encoding(self):
        return d.DEST_SYMBOL

    def __init__(self, *args, action_counts=0, **kwargs):
        super(Destination, self).__init__(*args, **kwargs)
        self.action_counts = action_counts
        self._per_agent_actions = defaultdict(lambda: 0)

    def do_wait_action(self, agent: Agent):
        self._per_agent_actions[agent.name] += 1
        return c.VALID

    @property
    def is_considered_reached(self):
        agent_at_position = any(c.AGENT.lower() in x.name.lower() for x in self.tile.guests_that_can_collide)
        return agent_at_position or any(x >= self.action_counts for x in self._per_agent_actions.values())

    def agent_did_action(self, agent: Agent):
        return self._per_agent_actions[agent.name] >= self.action_counts

    def summarize_state(self) -> dict:
        state_summary = super().summarize_state()
        state_summary.update(per_agent_times=[
            dict(belongs_to=key, time=val) for key, val in self._per_agent_actions.items()], counts=self.action_counts)
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
        return ((agent_at_position and not self.action_counts)
                or self._per_agent_actions[self.bound_entity.name] >= self.action_counts >= 1)
