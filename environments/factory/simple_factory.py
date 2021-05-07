import numpy as np
from environments.factory.base_factory import BaseFactory


class SimpleFactory(BaseFactory):
    def __init__(self, *args, max_dirt=5, **kwargs):
        self.max_dirt = max_dirt
        super(SimpleFactory, self).__init__(*args, **kwargs)

    def reset(self):
        super().reset()
        dirt_slice = np.zeros((1, *self.state.shape[1:]))
        self.state = np.concatenate((self.state, dirt_slice))  # dirt is now the last slice
        free_for_dirt = self.state.sum(0)
        free_for_dirt = np.argwhere(free_for_dirt == 0)
        np.random.shuffle(free_for_dirt)
        for x,y in free_for_dirt[:self.max_dirt]:
            self.state[-1, x, y] = 1
        print(self.state)


if __name__ == '__main__':
    factory = SimpleFactory(n_agents=1, max_dirt=8)
    #print(factory.state)
    state, r, done, _ = factory.step(0)
    #print(state)