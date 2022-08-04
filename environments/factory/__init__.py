def make(env_name, pomdp_r=2, max_steps=400, stack_n_frames=3, n_agents=1, individual_rewards=False):
    import yaml
    from pathlib import Path
    from environments.factory.combined_factories import DirtItemFactory
    from environments.factory.factory_item import ItemFactory
    from environments.factory.additional.item.item_util import ItemProperties
    from environments.factory.factory_dirt import DirtFactory
    from environments.factory.dirt_util import DirtProperties
    from environments.factory.dirt_util import RewardsDirt
    from environments.utility_classes import AgentRenderOptions

    with (Path(__file__).parent / 'levels' / 'parameters' / f'{env_name}.yaml').open('r') as stream:
        dictionary = yaml.load(stream, Loader=yaml.FullLoader)

    obs_props = dict(render_agents=AgentRenderOptions.COMBINED,
                     pomdp_r=pomdp_r,
                     indicate_door_area=True,
                     show_global_position_info=False,
                     frames_to_stack=stack_n_frames)

    factory_kwargs = dict(**dictionary,
                          n_agents=n_agents,
                          individual_rewards=individual_rewards,
                          max_steps=max_steps,
                          obs_prop=obs_props,
                          verbose=False,
                          )
    return DirtFactory(**factory_kwargs).__enter__()
