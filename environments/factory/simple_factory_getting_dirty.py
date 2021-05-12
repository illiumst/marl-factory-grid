import numpy as np
from environments.factory.base_factory import BaseFactory
from collections import namedtuple

DIRT_INDEX = -1
DirtProperties = namedtuple('DirtProperties', ['clean_amount', 'max_spawn_ratio', 'gain_amount'],
                            defaults=[0.25, 0.1, 0.1])


class GettingDirty(BaseFactory):

    def __init__(self, *args, dirt_properties:DirtProperties, **kwargs):
        super(GettingDirty, self).__init__(*args, **kwargs)
        self._dirt_properties = dirt_properties
        self.slice_strings.update({self.state.shape[0]-1: 'dirt'})

    def spawn_dirt(self):
        free_for_dirt = self.free_cells
        # randomly distribute dirt across the grid
        n_dirt_tiles = self._dirt_properties.max_spawn_ratio * len(free_for_dirt)
        for x, y in free_for_dirt[:n_dirt_tiles]:
            self.state[DIRT_INDEX, x, y] += self._dirt_properties.gain_amount

    def additional_actions(self, agent_i, action) -> ((int, int), bool):
        if action ==

    def reset(self):
        # ToDo: When self.reset returns the new states and stuff, use it here!
        super().reset()  # state, agents, ... =
        dirt_slice = np.zeros((1, *self.state.shape[1:]))
        self.state = np.concatenate((self.state, dirt_slice))  # dirt is now the last slice
        self.spawn_dirt()

    def calculate_reward(self, collisions_vecs, actions, r):
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
