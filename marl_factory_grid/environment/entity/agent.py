from typing import List, Union

from marl_factory_grid.environment.actions import Action
from marl_factory_grid.environment.entity.entity import Entity
from marl_factory_grid.utils.render import RenderEntity
from marl_factory_grid.utils import renderer
from marl_factory_grid.utils.helpers import is_move
from marl_factory_grid.utils.results import ActionResult, Result

from marl_factory_grid.environment import constants as c


class Agent(Entity):

    @property
    def var_is_blocking_light(self):
        return False

    @property
    def var_can_move(self):
        return True

    @property
    def var_is_blocking_pos(self):
        return False

    @property
    def var_has_position(self):
        return True

    @property
    def obs_tag(self):
        return self.name

    @property
    def actions(self):
        return self._actions

    @property
    def observations(self):
        return self._observations

    @property
    def var_can_collide(self):
        return True

    def step_result(self):
        pass

    @property
    def collection(self):
        return self._collection

    @property
    def state(self):
        return self._state or ActionResult(entity=self, identifier=c.NOOP, validity=c.VALID, reward=0)

    def __init__(self, actions: List[Action], observations: List[str], *args, **kwargs):
        super(Agent, self).__init__(*args, **kwargs)
        self.step_result = dict()
        self._actions = actions
        self._observations = observations
        self._state: Union[Result, None] = None

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
