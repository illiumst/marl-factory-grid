from typing import List, Union

import gym


class Entities():

    def __init__(self):
        pass


# noinspection PyAttributeOutsideInit
class BaseFactory(gym.Env):

    def __enter__(self):
        return self if self.frames_to_stack == 0 else FrameStack(self, self.frames_to_stack)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __init__(self, level_name='simple', n_agents=1, max_steps=int(5e2), pomdp_radius: Union[None, int] = 0,
                 movement_properties: MovementProperties = MovementProperties(),
                 combin_agent_slices_in_obs: bool = False, frames_to_stack=0,
                 omit_agent_slice_in_obs=False, **kwargs):
        assert (combin_agent_slices_in_obs != omit_agent_slice_in_obs) or \
               (not combin_agent_slices_in_obs and not omit_agent_slice_in_obs), \
            'Both options are exclusive'
        assert frames_to_stack != 1 and frames_to_stack >= 0, "'frames_to_stack' cannot be negative or 1."

        self.movement_properties = movement_properties
        self.level_name = level_name

        self.n_agents = n_agents
        self.max_steps = max_steps
        self.pomdp_radius = pomdp_radius
        self.combin_agent_slices_in_obs = combin_agent_slices_in_obs
        self.omit_agent_slice_in_obs = omit_agent_slice_in_obs
        self.frames_to_stack = frames_to_stack

        self.done_at_collision = False

        self._state_slices = StateSlices()
        level_filepath = Path(__file__).parent / h.LEVELS_DIR / f'{self.level_name}.txt'
        parsed_level = h.parse_level(level_filepath)
        self._level = h.one_hot_level(parsed_level)
        parsed_doors = h.one_hot_level(parsed_level, h.DOOR)
        if parsed_doors.any():
            self._doors = parsed_doors
            level_slices = ['level', 'doors']
            can_use_doors = True
        else:
            level_slices = ['level']
            can_use_doors = False
        offset = len(level_slices)
        self._state_slices.register_additional_items([*level_slices,
                                                      *[f'agent#{i}' for i in range(offset, n_agents + offset)]])
        if 'additional_slices' in kwargs:
            self._state_slices.register_additional_items(kwargs.get('additional_slices'))
        self._zones = Zones(parsed_level)

        self._actions = Actions(self.movement_properties, can_use_doors=can_use_doors)
        self._actions.register_additional_items(self.additional_actions)
        self.reset()


    def step(self, actions: Union[int, List[int]]):
        actions = actions if isinstance(actions, list) else [actions]
        self.entities.step()