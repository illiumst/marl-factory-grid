import os
import pickle
from pathlib import Path

from tqdm import trange

from marl_factory_grid import Factory
from marl_factory_grid.algorithms.static.contortions import get_coin_quadrant_tsp_agents, get_two_rooms_tsp_agents


def coin_quadrant_multi_agent_tsp_eval(emergent_phenomenon):
    run_tsp_setting("coin_quadrant", emergent_phenomenon, log=False)


def two_rooms_multi_agent_tsp_eval(emergent_phenomenon):
    run_tsp_setting("two_rooms", emergent_phenomenon, log=False)


def run_tsp_setting(config_name, emergent_phenomenon, n_episodes=1, log=False):
    # Render at each step?
    render = True

    # Path to config File
    path = Path(f'./marl_factory_grid/configs/tsp/{config_name}.yaml')

    # Create results folder
    runs = os.listdir("./study_out/")
    run_numbers = [int(run[7:]) for run in runs if run[:7] == "tsp_run"]
    next_run_number = max(run_numbers) + 1 if run_numbers else 0
    results_path = f"./study_out/tsp_run{next_run_number}"
    os.mkdir(results_path)

    # Env Init
    factory = Factory(path)

    with open(f"{results_path}/env_config.txt", "w") as txt_file:
        txt_file.write(str(factory.conf))

    still_existing_coin_piles = []
    reached_flags = []

    for episode in trange(n_episodes):
        _ = factory.reset()
        still_existing_coin_piles.append([])
        reached_flags.append([])
        done = False
        if render:
            factory.render()
            factory._renderer.fps = 5
        if config_name == "coin_quadrant":
            agents = get_coin_quadrant_tsp_agents(emergent_phenomenon, factory)
        elif config_name == "two_rooms":
            agents = get_two_rooms_tsp_agents(emergent_phenomenon, factory)
        else:
            print("Config name does not exist. Abort...")
            break
        ep_steps = 0
        while not done:
            a = [x.predict() for x in agents]
            # Have this condition, to terminate as soon as all coin piles are collected. This ensures that the implementation
            # of the TSP agent is equivalent to that of the RL agent
            if 'CoinPiles' in list(factory.state.entities.keys()) and factory.state.entities['CoinPiles'].global_amount == 0.0:
                break
            obs_type, _, _, done, info = factory.step(a)
            if 'CoinPiles' in list(factory.state.entities.keys()):
                still_existing_coin_piles[-1].append(len(factory.state.entities['CoinPiles']))
            if 'Destinations' in list(factory.state.entities.keys()):
                reached_flags[-1].append(sum([1 for ele in [x.was_reached() for x in factory.state['Destinations']] if ele]))
            ep_steps += 1
            if render:
                factory.render()
            if done:
                break

        collected_coin_piles_per_step = []
        if 'CoinPiles' in list(factory.state.entities.keys()):
            for ep in still_existing_coin_piles:
                collected_coin_piles_per_step.append([max(ep)-ep[idx] for idx, value in enumerate(ep)])
            # Remove first element and add last element where all coin piles have been collected
            del collected_coin_piles_per_step[-1][0]
            collected_coin_piles_per_step[-1].append(max(still_existing_coin_piles[-1]))

        # Add last entry to reached_flags
        print("Number of environment steps:", ep_steps)
        if 'CoinPiles' in list(factory.state.entities.keys()):
            print("Collected coins per step:", collected_coin_piles_per_step)
        if 'Destinations' in list(factory.state.entities.keys()):
            print("Reached flags per step:", reached_flags)

        if log:
            if 'CoinPiles' in list(factory.state.entities.keys()):
                metrics_data = {"collected_coin_piles_per_step": collected_coin_piles_per_step}
            if 'Destinations' in list(factory.state.entities.keys()):
                metrics_data = {"reached_flags": reached_flags}
            with open(f"{results_path}/metrics", "wb") as pickle_file:
                pickle.dump(metrics_data, pickle_file)