import random


# noinspection PyAbstractClass
from environments.factory.additional.btry.btry_util import BatteryProperties
from environments.factory.additional.btry.factory_battery import BatteryFactory
from environments.factory.additional.dest.factory_dest import DestFactory
from environments.factory.additional.dirt.dirt_util import DirtProperties
from environments.factory.additional.dirt.factory_dirt import DirtFactory
from environments.factory.additional.doors.factory_doors import DoorFactory
from environments.factory.additional.item.factory_item import ItemFactory


# noinspection PyAbstractClass
class DoorDirtFactory(DoorFactory, DirtFactory):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


# noinspection PyAbstractClass
class DirtItemFactory(ItemFactory, DirtFactory):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


# noinspection PyAbstractClass
class DirtBatteryFactory(DirtFactory, BatteryFactory):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


# noinspection PyAbstractClass
class DirtDestItemFactory(ItemFactory, DirtFactory, DestFactory):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


# noinspection PyAbstractClass
class DestBatteryFactory(BatteryFactory, DestFactory):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


if __name__ == '__main__':
    from environments.utility_classes import AgentRenderOptions as ARO, ObservationProperties

    render = True

    obs_props = ObservationProperties(render_agents=ARO.COMBINED, omit_agent_self=True,
                                      pomdp_r=2, additional_agent_placeholder=None)

    move_props = {'allow_square_movement': True,
                  'allow_diagonal_movement': False,
                  'allow_no_op': False}

    factory = DoorDirtFactory(n_agents=10, done_at_collision=False,
                                 level_name='rooms', max_steps=400,
                                 obs_prop=obs_props, parse_doors=True,
                                 record_episodes=True, verbose=True,
                                 dirt_prop=DirtProperties(),
                                 mv_prop=move_props)


    # noinspection DuplicatedCode
    n_actions = factory.action_space.n - 1
    _ = factory.observation_space

    for epoch in range(4):
        random_actions = [[random.randint(0, n_actions) for _
                           in range(factory.n_agents)] for _
                          in range(factory.max_steps + 1)]
        env_state = factory.reset()
        r = 0
        for agent_i_action in random_actions:
            env_state, step_r, done_bool, info_obj = factory.step(agent_i_action)
            r += step_r
            if render:
                factory.render()
            if done_bool:
                break
        print(f'Factory run {epoch} done, reward is:\n    {r}')
pass
