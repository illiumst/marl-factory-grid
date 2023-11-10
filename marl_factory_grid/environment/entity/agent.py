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
        return len(self._paralyzed)

    @property
    def paralyze_reasons(self):
        return [x for x in self._paralyzed]

    @property
    def obs_tag(self):
        return self.name

    @property
    def actions(self):
        return self._actions

    @property
    def observations(self):
        return self._observations

    def step_result(self):
        pass

    @property
    def collection(self):
        return self._collection

    @property
    def var_is_blocking_pos(self):
        return self._is_blocking_pos

    @property
    def state(self):
        return self._state or ActionResult(entity=self, identifier=c.NOOP, validity=c.VALID)

    def __init__(self, actions: List[Action], observations: List[str], *args, is_blocking_pos=False, **kwargs):
        super(Agent, self).__init__(*args, **kwargs)
        self._paralyzed = set()
        self.step_result = dict()
        self._actions = actions
        self._observations = observations
        self._state: Union[Result, None] = None
        self._is_blocking_pos = is_blocking_pos

    # noinspection PyAttributeOutsideInit
    def clear_temp_state(self):
        self._state = None
        return self

    def summarize_state(self):
        state_dict = super().summarize_state()
        state_dict.update(valid=bool(self.state.validity), action=str(self.state.identifier))
        return state_dict

    def set_state(self, action_result):
        self._state = action_result

    def paralyze(self, reason):
        self._paralyzed.add(reason)
        return c.VALID

    def de_paralyze(self, reason):
        try:
            self._paralyzed.remove(reason)
            return c.VALID
        except KeyError:
            return c.NOT_VALID

    def render(self):
        i = next(idx for idx, x in enumerate(self._collection) if x.name == self.name)
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
