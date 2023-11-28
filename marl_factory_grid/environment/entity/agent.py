from typing import List, Union

from marl_factory_grid.environment.actions import Action
from marl_factory_grid.environment.entity.entity import Entity
from marl_factory_grid.utils.utility_classes import RenderEntity
from marl_factory_grid.utils import renderer
from marl_factory_grid.utils.helpers import is_move
from marl_factory_grid.utils.results import ActionResult, Result

from marl_factory_grid.environment import constants as c


class Agent(Entity):

    @property
    def var_is_paralyzed(self) -> bool:
        """
        Check if the Agent is able to move and perform actions. Can be paralized by eg. damage or empty battery.

        :return:  Wether the Agent is paralyzed.
        """
        return bool(len(self._paralyzed))

    @property
    def paralyze_reasons(self) -> list[str]:
        """
        Reveals the reasons for the recent paralyzation.

        :return: A list of strings.
        """
        return [x for x in self._paralyzed]

    @property
    def obs_tag(self):
        """Internal Usage"""
        return self.name

    @property
    def actions(self):
        """
        Reveals the actions this agent is capable of.

        :return: List of actions.
        """
        return self._actions

    @property
    def observations(self):
        """
        Reveals the observations which this agent wants to see.

        :return: List of observations.
        """
        return self._observations

    @property
    def var_is_blocking_pos(self):
        return self._is_blocking_pos

    def __init__(self, actions: List[Action], observations: List[str], *args, is_blocking_pos=False, **kwargs):
        """
        This is the main agent surrogate.
        Actions given to env.step() are associated with this entity and performed at `on_step`.


        :param kwargs: object
        :param args: object
        :param is_blocking_pos: object
        :param observations: object
        :param actions: object
        """
        super(Agent, self).__init__(*args, **kwargs)
        self._paralyzed = set()
        self.step_result = dict()
        self._actions = actions
        self._observations = observations
        self._status: Union[Result, None] = None
        self._is_blocking_pos = is_blocking_pos

    def summarize_state(self) -> dict[str]:
        """
        More or less the result of the last action. Usefull for debugging and used in renderer.

        :return: Last action result
        """
        state_dict = super().summarize_state()
        state_dict.update(valid=bool(self.state.validity), action=str(self.state.identifier))
        return state_dict

    def set_state(self, state: Result) -> bool:
        """
        Place result in temp agent state.

        :return: Always true
        """
        self._status = state
        return c.VALID

    def paralyze(self, reason):
        """
        Paralyze an agent. Paralyzed agents are not able to do actions.
        This is usefull, when battery is empty or agent is damaged.

        :return: Always true
        """
        self._paralyzed.add(reason)
        return c.VALID

    def de_paralyze(self, reason) -> bool:
        """
        De-paralyze an agent, so that he is able to perform actions again.

        :return:
        """
        try:
            self._paralyzed.remove(reason)
            return c.VALID
        except KeyError:
            return c.NOT_VALID

    def render(self) -> RenderEntity:
        i = self.collection.idx_by_entity(self)
        assert i is not None
        curr_state = self.state
        name = c.AGENT
        if curr_state.identifier == c.COLLISION:
            name = renderer.STATE_COLLISION
            render_state=None
        elif curr_state.validity:
            if curr_state.identifier == c.NOOP:
                render_state = renderer.STATE_IDLE
            elif is_move(curr_state.identifier):
                render_state = renderer.STATE_MOVE
            else:
                render_state = renderer.STATE_VALID
        else:
            render_state = renderer.STATE_INVALID

        return RenderEntity(name, self.pos, 1, 'none', render_state, i + 1, real_name=self.name)
