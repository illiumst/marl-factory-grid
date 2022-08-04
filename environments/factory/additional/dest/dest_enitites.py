from collections import defaultdict

from environments.factory.base.objects import Entity, Agent
from environments.factory.additional.dest.dest_util import Constants as c


class Destination(Entity):

    @property
    def any_agent_has_dwelled(self):
        return bool(len(self._per_agent_times))

    @property
    def currently_dwelling_names(self):
        return self._per_agent_times.keys()

    @property
    def encoding(self):
        return c.DESTINATION

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

    def summarize_state(self, n_steps=None) -> dict:
        state_summary = super().summarize_state(n_steps=n_steps)
        state_summary.update(per_agent_times=self._per_agent_times)
        return state_summary
