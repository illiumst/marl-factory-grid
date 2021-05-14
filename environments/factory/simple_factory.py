import numpy as np
from environments.factory.base_factory import BaseFactory, FactoryMonitor


class SimpleFactory(BaseFactory):
    def __init__(self, *args, max_dirt=5, **kwargs):
        self.max_dirt = max_dirt
        super(SimpleFactory, self).__init__(*args, **kwargs)
        self.slice_strings.update({self.state.shape[0]-1: 'dirt'})

    def spawn_dirt(self):
        free_for_dirt = self.free_cells
        for x, y in free_for_dirt[:self.max_dirt]:  # randomly distribute dirt across the grid
            self.state[-1, x, y] = 1

    def reset(self):
        state, r, done, _ = super().reset()
        dirt_slice = np.zeros((1, *self.state.shape[1:]))
        self.state = np.concatenate((self.state, dirt_slice))  # dirt is now the last slice
        self.spawn_dirt()
        # Always: This should return state, r, done, info
        return self.state, r, done, _

    def calculate_reward(self, agent_states):
        for agent_state in agent_states:
            collisions = agent_state.collisions
            entities = [self.slice_strings[entity] for entity in collisions]
            if entities:
                for entity in entities:
                    self.monitor.add(f'agent_{agent_state.i}_collision_{entity}', 1)
                print(f't = {self.steps}\tAgent {agent_state.i} has collisions with '
                      f'{entities}')
        return 0, {}


if __name__ == '__main__':
    import random
    factory = SimpleFactory(n_agents=1, max_dirt=8)
    monitor_list = list()
    for epoch in range(100):
        random_actions = [random.randint(0, 7) for _ in range(200)]
        state, r, done, _ = factory.reset()
        for action in random_actions:
            state, r, done, info = factory.step(action)
        monitor_list.append(factory.monitor)

        print(f'Factory run done, reward is:\n    {r}')
        print(f'There have been the following collisions: \n {dict(factory.monitor)}')

