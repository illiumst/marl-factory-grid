from pathlib import Path
from random import randint

from tqdm import trange

from marl_factory_grid.environment.factory import Factory

if __name__ == '__main__':
    # Render at each step?
    render = False

    # Path to config File
    path = Path('test_config.yaml')

    # Env Init
    factory = Factory(path)

    for episode in trange(5):
        _ = factory.reset()
        done = False
        if render:
            factory.render()
        action_spaces = factory.action_space
        while not done:
            a = [randint(0, x.n - 1) for x in action_spaces]
            obs_type, _, _, done, info = factory.step(a)
            if render:
                factory.render()
            if done:
                print(f'Episode {episode} done...')
                break

