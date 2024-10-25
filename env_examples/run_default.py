from pathlib import Path
from tqdm import trange
from marl_factory_grid.algorithms.static.TSP_coin_agent import TSPCoinAgent
from marl_factory_grid.algorithms.static.TSP_dirt_agent import TSPDirtAgent
from marl_factory_grid.environment.factory import Factory

if __name__ == '__main__':

    run_path = Path('../study_out')
    render = True
    monitor = True
    record = True

    # Path to config File
    path = Path('../marl_factory_grid/configs/default_config.yaml')

    # Env Init
    factory = Factory(path)

    for episode in trange(10):
        _ = factory.reset()
        done = False
        if render:
            factory.render()
        action_spaces = factory.action_space
        agents = [TSPDirtAgent(factory, 0), TSPCoinAgent(factory, 1)]
        while not done:
            a = [x.predict() for x in agents]
            obs_type, _, _, done, info = factory.step(a)
            if render:
                factory.render()
            if done:
                print(f'Episode {episode} done...')
                break
