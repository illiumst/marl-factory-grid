def rooms(n_agents=1):
    from environments.factory.factory_dirt_item import DirtItemFactory
    from environments.factory.factory_item import ItemFactory, ItemProperties
    from environments.factory.factory_dirt import DirtProperties, DirtFactory
    from environments.utility_classes import MovementProperties, ObservationProperties, AgentRenderOptions

    obs_props = ObservationProperties(render_agents=AgentRenderOptions.NOT,
                                      omit_agent_self=True,
                                      additional_agent_placeholder=None,
                                      frames_to_stack=0,
                                      pomdp_r=2
                                      )
    move_props = MovementProperties(allow_diagonal_movement=True,
                                    allow_square_movement=True,
                                    allow_no_op=False)
    dirt_props = DirtProperties(initial_dirt_ratio=0.35, initial_dirt_spawn_r_var=0.1,
                                clean_amount=0.34,
                                max_spawn_amount=0.1, max_global_amount=20,
                                max_local_amount=1, spawn_frequency=0, max_spawn_ratio=0.05,
                                dirt_smear_amount=0.0, agent_can_interact=True)
    factory_kwargs = dict(n_agents=n_agents, max_steps=400, parse_doors=True,
                          level_name='rooms', record_episodes=False, doors_have_area=False,
                          verbose=False,
                          mv_prop=move_props,
                          obs_prop=obs_props
                          )
    return DirtFactory(dirt_props=dirt_props, **factory_kwargs)
