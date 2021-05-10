import numpy as np
from environments.factory.base_factory import BaseFactory


class SimpleFactory(BaseFactory):
    def __init__(self, *args, max_dirt=5, **kwargs):
        self.max_dirt = max_dirt
        super(SimpleFactory, self).__init__(*args, **kwargs)
        self.slice_strings.update({self.state.shape[0]-1: 'dirt'})

    def reset(self):
        super().reset()
        dirt_slice = np.zeros((1, *self.state.shape[1:]))
        self.state = np.concatenate((self.state, dirt_slice))  # dirt is now the last slice
        free_for_dirt = self.free_cells()
        for x, y in free_for_dirt[:self.max_dirt]:
            self.state[-1, x, y] = 1

    def step_core(self, collisions_vecs, actions, r):
        for agent_i, cols in enumerate(collisions_vecs):
            cols = np.argwhere(cols != 0).flatten()
            print(f'Agent #{agent_i} has collisions with '
                  f'{[self.slice_strings[entity] for entity in cols]}')
        return 0, {}



if __name__ == '__main__':
    import random
    factory = SimpleFactory(n_agents=1, max_dirt=8)
    random_actions = [random.randint(0,8) for _ in range(200)]
    for action in random_actions:
        state, r, done, _ = factory.step(action)