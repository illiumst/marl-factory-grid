from pathlib import Path
from random import randint

from tqdm import trange

from marl_factory_grid.algorithms.static.TSP_dirt_agent import TSPDirtAgent
from marl_factory_grid.environment.factory import Factory

if __name__ == '__main__':
    # Render at each step?
    render = True

    # Path to config File
    path = Path('marl_factory_grid/configs/test_config.yaml')

    # Env Init
    factory = Factory(path)

    for episode in trange(5):
        _ = factory.reset()
        done = False
        if render:
            factory.render()
        action_spaces = factory.action_space
        agents = [TSPDirtAgent(factory, 0)]
        while not done:
            a = [randint(0, x.n - 1) for x in action_spaces]
            obs_type, _, _, done, info = factory.step(a)
            if render:
                factory.render()
            if done:
                print(f'Episode {episode} done...')
                break

