import numpy as np
from environments.factory.base_factory import BaseFactory
from collections import namedtuple


DirtProperties = namedtuple('DirtProperties', ['clean_amount', 'max_spawn_ratio', 'gain_amount'])


class GettingDirty(BaseFactory):

    _dirt_indx = -1

    def __init__(self, *args, dirt_properties, **kwargs):
        super(GettingDirty, self).__init__(*args, **kwargs)
        self._dirt_properties = dirt_properties
        self.slice_strings.update({self.state.shape[0]-1: 'dirt'})

    def spawn_dirt(self):
        free_for_dirt = self.free_cells
        for x, y in free_for_dirt[:self._max_dirt_spawn_ratio * free_for_dirt.]:  # randomly distribute dirt across the grid
            self.state[self._dirt_indx, x, y] += 0.1

    def reset(self):
        # ToDo: When self.reset returns the new states and stuff, use it here!
        super().reset()  # state, agents, ... =
        dirt_slice = np.zeros((1, *self.state.shape[1:]))
        self.state = np.concatenate((self.state, dirt_slice))  # dirt is now the last slice
        self.spawn_dirt()

    def step_core(self, collisions_vecs, actions, r):
        for agent_i, cols in enumerate(collisions_vecs):
            cols = np.argwhere(cols != 0).flatten()
            print(f't = {self.steps}\tAgent {agent_i} has collisions with '
                  f'{[self.slice_strings[entity] for entity in cols]}')
        return 0, {}


if __name__ == '__main__':
    import random
    factory = GettingDirty(n_agents=1, max_dirt=8)
    random_actions = [random.randint(0, 8) for _ in range(200)]
    for action in random_actions:
        state, r, done, _ = factory.step(action)
