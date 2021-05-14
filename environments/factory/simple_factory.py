import numpy as np
from environments.factory.base_factory import BaseFactory


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
        super().reset()
        dirt_slice = np.zeros((1, *self.state.shape[1:]))
        self.state = np.concatenate((self.state, dirt_slice))  # dirt is now the last slice
        self.spawn_dirt()

    def calculate_reward(self, agent_states):
        for agent_state in agent_states:
            collisions = agent_state.collisions
            print(f't = {self.steps}\tAgent {agent_state.i} has collisions with '
                  f'{[self.slice_strings[entity] for entity in collisions]}')
        return 0, {}


if __name__ == '__main__':
    import random
    factory = SimpleFactory(n_agents=1, max_dirt=8)
    random_actions = [random.randint(0, 7) for _ in range(200)]
    for action in random_actions:
        state, r, done, _ = factory.step(action)
