import numpy as np
from pathlib import Path

# Constants
WALL = '#'


# Utility functions
def parse_level(path):
    with path.open('r') as lvl:
        level = list(map(lambda x: list(x.strip()), lvl.readlines()))
    if len(set([len(line) for line in level])) > 1:
        raise AssertionError('Every row of the level string must be of equal length.')
    return level


def one_hot_level(level, wall_char=WALL):
    grid = np.array(level)
    binary_grid = np.zeros(grid.shape)
    binary_grid[grid == wall_char] = 1
    return binary_grid


def check_agent_move(state, dim, action):
    agent_slice = state[dim]  # horizontal slice from state tensor
    agent_pos = np.argwhere(agent_slice == 1)
    if len(agent_pos) > 1:
        raise AssertionError('Only one agent per slice is allowed.')
    x, y = agent_pos[0]
    x_new, y_new = x, y
    # Actions
    if action == 0: # North
        x_new -= 1
    elif action == 1: # East
        y_new += 1
    elif action == 2: # South
        x_new += 1
    elif action == 3: # West
        y_new -= 1
    elif action == 4: # NE
        x_new -= 1
        y_new += 1
    elif action == 5: # SE
        x_new += 1
        y_new += 1
    elif action == 6: # SW
        x_new += 1
        y_new -= 1
    elif action == 7: # NW
        x_new -= 1
        y_new -= 1
    # Check validity
    valid = not (
            x_new < 0 or y_new < 0
            or x_new >= agent_slice.shape[0]
            or y_new >= agent_slice.shape[0]
    )  # if agent tried to leave the grid
    return (x, y), (x_new, y_new), valid





if __name__ == '__main__':
    x = parse_level(Path(__file__).parent / 'factory' / 'levels' / 'simple.txt')
    y = one_hot_level(x)
    print(np.argwhere(y == 0))