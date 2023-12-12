Using the environment with your agents
===========================================

Environment objects, including agents, entities and rules, that are specified in a *yaml*-configfile will be loaded automatically.
Using ``quickstart_use`` creates a default config-file and another one that lists all possible options of the environment.
Also, it generates an initial script where an agent is executed in the environment specified by the config-file.

After initializing the environment using the specified configuration file, the script enters a reinforcement learning loop.
The loop consists of episodes, where each episode involves resetting the environment, executing actions, and receiving feedback.

Here's a breakdown of the key components in the provided script. Feel free to customize it based on your specific requirements:

1. **Initialization:**

>>> path = Path('marl_factory_grid/configs/default_config.yaml')
    factory = Factory(path)
    factory = EnvMonitor(factory)
    factory = EnvRecorder(factory)

    - The `path` variable points to the location of your configuration file. Ensure it corresponds to the correct path.
    - `Factory` initializes the environment based on the provided configuration.
    - `EnvMonitor` and `EnvRecorder` are optional components. They add monitoring and recording functionalities to the environment, respectively.

2. **Reinforcement Learning Loop:**

>>> for episode in trange(10):
        _ = factory.reset()
        done = False
        if render:
            factory.render()
        action_spaces = factory.action_space
        agents = []

    - The loop iterates over a specified number of episodes (in this case, 10).
    - `factory.reset()` resets the environment for a new episode.
    - `factory.render()` is used for visualization if rendering is enabled.
    - `action_spaces` stores the action spaces available for the agents.
    - `agents` will store agent-specific information during the episode.

3. **Taking Actions:**

>>> while not done:
        a = [randint(0, x.n - 1) for x in action_spaces]
        obs_type, _, reward, done, info = factory.step(a)
        if render:
            factory.render()

    - Within each episode, the loop continues until the environment signals completion (`done`).
    - `a` represents a list of random actions for each agent based on their action space.
    - `factory.step(a)` executes the actions, returning observation types, rewards, completion status, and additional information.

4. **Handling Episode Completion:**

>>> if done:
        print(f'Episode {episode} done...')

    - After each episode, a message is printed indicating its completion.


Evaluating the run
----

If monitoring and recording are enabled, the environment states will be traced and recorded automatically.

Plotting. At the moment a plot of the evaluation score across the different episodes is automatically generated.
