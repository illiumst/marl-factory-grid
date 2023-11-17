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
    def var_is_paralyzed(self):
        """
        TODO


        :return:
        """
        return len(self._paralyzed)

    @property
    def paralyze_reasons(self):
        """
        TODO


        :return:
        """
        return [x for x in self._paralyzed]

    @property
    def obs_tag(self):
        """Internal Usage"""
        return self.name

    @property
    def actions(self):
        """
        TODO


        :return:
        """
        return self._actions

    @property
    def observations(self):
        """
        TODO


        :return:
        """
        return self._observations

    def step_result(self):
        """
        TODO
        FIXME THINK ITS LEGACY... Not Used any more


        :return:
        """
        pass

    @property
    def var_is_blocking_pos(self):
        return self._is_blocking_pos

    def __init__(self, actions: List[Action], observations: List[str], *args, is_blocking_pos=False, **kwargs):
        """
        TODO


        :return:
        """
        super(Agent, self).__init__(*args, **kwargs)
        self._paralyzed = set()
        self.step_result = dict()
        self._actions = actions
        self._observations = observations
        self._status: Union[Result, None] = None
        self._is_blocking_pos = is_blocking_pos

    def summarize_state(self):
        """
        TODO


        :return:
        """
        state_dict = super().summarize_state()
        state_dict.update(valid=bool(self.state.validity), action=str(self.state.identifier))
        return state_dict

    def set_state(self, state):
        """
        TODO


        :return:
        """
        self._status = state
        return c.VALID


    def paralyze(self, reason):
        """
        TODO


        :return:
        """
        self._paralyzed.add(reason)
        return c.VALID

    def de_paralyze(self, reason) -> bool:
        """
        TODO


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
        if curr_state.identifier == c.COLLISION:
            render_state = renderer.STATE_COLLISION
        elif curr_state.validity:
            if curr_state.identifier == c.NOOP:
                render_state = renderer.STATE_IDLE
            elif is_move(curr_state.identifier):
                render_state = renderer.STATE_MOVE
            else:
                render_state = renderer.STATE_VALID
        else:
            render_state = renderer.STATE_INVALID

        return RenderEntity(c.AGENT, self.pos, 1, 'none', render_state, i + 1, real_name=self.name)
