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
    # Reveal all possible Modules (Entities, Rules, Agents[Actions, Observations], etc.)
    explain_config = False
    # Collect statistics?
    monitor = True
    # Record as Protobuf?
    record = False
    # Plot Results?
    plotting = True

    run_path = Path('study_out')

    if explain_config:
        ce = ConfigExplainer()
        ce.save_all(run_path / 'all_available_configs.yaml')

    # Path to config File
    path = Path('marl_factory_grid/configs/eight_puzzle.yaml')

    # Env Init
    factory = Factory(path)

    # Record and Monitor
    if monitor:
        factory = EnvMonitor(factory)
    if record:
        factory = EnvRecorder(factory)

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

    if monitor:
        factory.save_run(run_path / 'test_monitor.pkl')
    if record:
        factory.save_records(run_path / 'test.pb')
    if plotting:
        factory.report_possible_colum_keys()
        plot_single_run(run_path, column_keys=['Global_DoneAtDestinationReachAll', 'step_reward',
                                               'Agent[Karl-Heinz]_DoneAtDestinationReachAll',
                                               'Agent[Wolfgang]_DoneAtDestinationReachAll',
                                               'Global_DoneAtDestinationReachAll'])

    print('Done!!! Goodbye....')
