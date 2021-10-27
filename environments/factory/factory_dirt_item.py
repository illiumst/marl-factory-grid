import random
from pathlib import Path

from environments.factory.factory_dirt import DirtFactory, DirtProperties
from environments.factory.factory_item import ItemFactory, ItemProperties
from environments.logging.recorder import RecorderCallback
from environments.utility_classes import MovementProperties


class DirtItemFactory(ItemFactory, DirtFactory):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


if __name__ == '__main__':
    with RecorderCallback(filepath=Path('debug_out') / f'recorder_xxxx.json', occupation_map=False,
                          trajectory_map=False) as recorder:

        dirt_props = DirtProperties(clean_amount=2, gain_amount=0.1, max_global_amount=20,
                                    max_local_amount=1, spawn_frequency=3, max_spawn_ratio=0.05,
                                    dirt_smear_amount=0.0, agent_can_interact=True)
        item_props = ItemProperties(n_items=5, agent_can_interact=True)
        move_props = MovementProperties(allow_diagonal_movement=True,
                                        allow_square_movement=True,
                                        allow_no_op=False)

        render = True

        factory = DirtItemFactory(n_agents=1, done_at_collision=False, frames_to_stack=0,
                              level_name='rooms', max_steps=200, combin_agent_obs=True,
                              omit_agent_in_obs=True, parse_doors=True, pomdp_r=3,
                              record_episodes=True, verbose=False, cast_shadows=True,
                              movement_properties=move_props, dirt_properties=dirt_props
                              )

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
                # recorder.read_info(0, info_obj)
                r += step_r
                if render:
                    factory.render()
                if done_bool:
                    # recorder.read_done(0, done_bool)
                    break
            print(f'Factory run {epoch} done, reward is:\n    {r}')
        pass
