import numpy as np
from pathlib import Path
from environments import helpers as h


class Factory(object):
    LEVELS_DIR = 'levels'

    def __init__(self, level='simple', n_agents=1, max_steps=1e3):
        self.n_agents = n_agents
        self.max_steps = max_steps
        self.level = h.one_hot_level(
            h.parse_level(Path(__file__).parent / self.LEVELS_DIR / f'{level}.txt')
        )#[np.newaxis, ...]
        self.reset()

    def reset(self):
        self.done = False
        self.agents = np.zeros((self.n_agents, *self.level.shape))
        free_cells = np.argwhere(self.level == 0)
        np.random.shuffle(free_cells)
        for i in range(self.n_agents):
            r, c = free_cells[i]
            self.agents[i, r, c] = 1
        free_cells = free_cells[self.n_agents:]
        self.state = np.concatenate((self.level[np.newaxis, ...], self.agents), 0)

    def step(self, actions):
        assert type(actions) in [int, list]
        if type(actions) == int:
            actions = [actions]
        # level, agent 1,..., agent n,
        for i, a in enumerate(actions):
            old_pos, new_pos, valid = h.check_agent_move(state=self.state, dim=i+1, action=a)
            print(old_pos, new_pos, valid)


if __name__ == '__main__':
    factory = Factory(n_agents=1)
    factory.step(0)