Using the environment with your agents
===========================================

Environment objects, including agents, entities and rules, that are specified in a *yaml*-configfile will be loaded automatically.
Using ``quickstart_use`` creates a default config-file and another one that lists all possible options of the environment.
Also, it generates an initial script where an agent is executed in the environment specified by the config-file.

The script initializes the environment, monitoring and recording of the environment, and includes the reinforcement learning loop:

>>>     path = Path('marl_factory_grid/configs/default_config.yaml')
        factory = Factory(path)
        factory = EnvMonitor(factory)
        factory = EnvRecorder(factory)
        for episode in trange(10):
            _ = factory.reset()
            done = False
            if render:
                factory.render()
            action_spaces = factory.action_space
            agents = []
            while not done:
                a = [randint(0, x.n - 1) for x in action_spaces]
                obs_type, _, reward, done, info = factory.step(a)
                if render:
                    factory.render()
                if done:
                    print(f'Episode {episode} done...')
                    break
