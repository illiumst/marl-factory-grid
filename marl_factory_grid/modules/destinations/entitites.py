from collections import defaultdict

from marl_factory_grid.environment import constants as c
from marl_factory_grid.environment.entity.agent import Agent
from marl_factory_grid.environment.entity.entity import Entity
from marl_factory_grid.modules.destinations import constants as d
from marl_factory_grid.utils.utility_classes import RenderEntity


class Destination(Entity):

    @property
    def encoding(self):
        return d.DEST_SYMBOL

    def __init__(self, *args, action_counts=0, **kwargs):
        """
        Represents a destination in the environment that agents aim to reach.

        """
        super(Destination, self).__init__(*args, **kwargs)
        self._was_reached = False
        self.action_counts = action_counts
        self._per_agent_actions = defaultdict(lambda: 0)

    def do_wait_action(self, agent) -> bool:
        """
        Performs a wait action for the given agent at the destination.

        :param agent: The agent performing the wait action.
        :type agent: Agent

        :return: Whether the action was valid or not.
        :rtype: bool
        """
        self._per_agent_actions[agent.name] += 1
        return c.VALID

    def has_just_been_reached(self, state):
        """
        Checks if the destination has just been reached based on the current state.
        """
        if self.was_reached():
            return False
        agent_at_position = any(state[c.AGENT].by_pos(self.pos))

        if self.bound_entity:
            return ((agent_at_position and not self.action_counts)
                    or self._per_agent_actions[self.bound_entity.name] >= self.action_counts >= 1)
        else:
            return agent_at_position or any(x >= self.action_counts for x in self._per_agent_actions.values())

    def agent_did_action(self, agent: Agent):
        """
        Internal usage, currently no usage.
        """
        return self._per_agent_actions[agent.name] >= self.action_counts

    def summarize_state(self) -> dict:
        state_summary = super().summarize_state()
        state_summary.update(per_agent_times=[
            dict(belongs_to=key, time=val) for key, val in self._per_agent_actions.items()], counts=self.action_counts)
        return state_summary

    def render(self):
        if self.was_reached():
            return None
        else:
            return RenderEntity(d.DESTINATION, self.pos)

    def mark_as_reached(self):
        self._was_reached = True

    def unmark_as_reached(self):
        self._was_reached = False

    def was_reached(self) -> bool:
        return self._was_reached
