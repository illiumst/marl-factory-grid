from pathlib import Path

from tqdm import trange

from marl_factory_grid.algorithms.static.TSP_dirt_agent import TSPDirtAgent
from marl_factory_grid.algorithms.static.TSP_item_agent import TSPItemAgent
from marl_factory_grid.algorithms.static.TSP_target_agent import TSPTargetAgent
from marl_factory_grid.environment.factory import Factory

if __name__ == '__main__':
    # Render at each step?
    render = False

    # Path to config File
    path = Path('marl_factory_grid/configs/test_config.yaml')

    # Env Init
    factory = Factory(path)

    for episode in trange(10):
        _ = factory.reset()
        done = False
        if render:
            factory.render()
        action_spaces = factory.action_space
        agents = [TSPDirtAgent(factory, 0), TSPItemAgent(factory, 1), TSPTargetAgent(factory, 2)]
        while not done:
            a = [x.predict() for x in agents]
            obs_type, _, _, done, info = factory.step(a)
            if render:
                factory.render()
            if done:
                print(f'Episode {episode} done...')
                break
