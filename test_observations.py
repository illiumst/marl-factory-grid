from pathlib import Path
from random import randint
from tqdm import trange

from marl_factory_grid.environment.factory import Factory

from marl_factory_grid.utils.logging.envmonitor import EnvMonitor
from marl_factory_grid.utils.logging.recorder import EnvRecorder
from marl_factory_grid.utils.plotting.plot_single_runs import plot_single_run
from marl_factory_grid.utils.tools import ConfigExplainer


if __name__ == '__main__':
    # Render at each step?
    render = True

    run_path = Path('study_out')

    # Path to config File
    path = Path('marl_factory_grid/configs/_obs_test.yaml')

    # Env Init
    factory = Factory(path)

    # RL learn Loop
    for episode in trange(10):
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

    print('Done!!! Goodbye....')
