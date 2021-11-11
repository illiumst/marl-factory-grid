def make(env_str, n_agents=1, pomdp_r=2, max_steps=400):
    import yaml
    from pathlib import Path
    from environments.factory.combined_factories import DirtItemFactory
    from environments.factory.factory_item import ItemFactory, ItemProperties
    from environments.factory.factory_dirt import DirtProperties, DirtFactory
    from environments.utility_classes import MovementProperties, ObservationProperties, AgentRenderOptions

    with (Path(__file__).parent / 'levels' / 'parameters' / f'{env_str}.yaml').open('r') as stream:
        dictionary = yaml.load(stream, Loader=yaml.FullLoader)

    obs_props = ObservationProperties(render_agents=AgentRenderOptions.COMBINED, frames_to_stack=0, pomdp_r=pomdp_r)

    factory_kwargs = dict(n_agents=n_agents, max_steps=max_steps, obs_prop=obs_props,
                          mv_prop=MovementProperties(**dictionary['movement_props']),
                          dirt_prop=DirtProperties(**dictionary['dirt_props']),
                          record_episodes=False, verbose=False, **dictionary['factory_props']
                          )

    return DirtFactory(**factory_kwargs)
