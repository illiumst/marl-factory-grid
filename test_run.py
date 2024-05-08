from pathlib import Path
from pprint import pprint

from tqdm import trange

from marl_factory_grid.algorithms.static.TSP_dirt_agent import TSPDirtAgent
from marl_factory_grid.algorithms.static.TSP_item_agent import TSPItemAgent
from marl_factory_grid.algorithms.static.TSP_target_agent import TSPTargetAgent
from marl_factory_grid.environment.factory import Factory

from marl_factory_grid.utils.plotting.plot_single_runs import plot_routes, plot_action_maps

if __name__ == '__main__':

    run_path = Path('study_out')
    render = True
    monitor = True
    record = True

    # Path to config File
    path = Path('marl_factory_grid/configs/test_config.yaml')

    # Env Init
    factory = Factory(path)

    for episode in trange(1):
        _ = factory.reset()
        done = False
        if render:
            factory.render()
        action_spaces = factory.action_space
        # agents = [TSPDirtAgent(factory, 0), TSPItemAgent(factory, 1), TSPTargetAgent(factory, 2)]
        agents = [TSPTargetAgent(factory, 0), TSPTargetAgent(factory, 1)]
        while not done:
            a = [x.predict() for x in agents]
            obs_type, _, _, done, info = factory.step(a)
            if render:
                factory.render()
            if done:
                print(f'Episode {episode} done...')
                break

        plot_action_maps(factory, agents)
