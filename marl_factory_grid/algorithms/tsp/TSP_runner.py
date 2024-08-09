import os
import pickle
from pathlib import Path

from tqdm import trange

from marl_factory_grid import Factory
from marl_factory_grid.algorithms.tsp.contortions import get_dirt_quadrant_tsp_agents, get_two_rooms_tsp_agents


def dirt_quadrant_multi_agent_tsp_eval(emergent_phenomenon):
    run_tsp_setting("dirt_quadrant", emergent_phenomenon, log=False)


def two_rooms_multi_agent_tsp_eval(emergent_phenomenon):
    run_tsp_setting("two_rooms", emergent_phenomenon, log=False)


def run_tsp_setting(config_name, emergent_phenomenon, n_episodes=1, log=False):
    # Render at each step?
    render = True

    # Path to config File
    path = Path(f'./marl_factory_grid/environment/configs/tsp/{config_name}.yaml')

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

    still_existing_dirt_piles = []
    reached_flags = []

    for episode in trange(n_episodes):
        _ = factory.reset()
        still_existing_dirt_piles.append([])
        reached_flags.append([])
        done = False
        if render:
            factory.render()
            factory._renderer.fps = 5
        if config_name == "dirt_quadrant":
            agents = get_dirt_quadrant_tsp_agents(emergent_phenomenon, factory)
        elif config_name == "two_rooms":
            agents = get_two_rooms_tsp_agents(emergent_phenomenon, factory)
        else:
            print("Config name does not exist. Abort...")
            break
        ep_steps = 0
        while not done:
            a = [x.predict() for x in agents]
            # Have this condition, to terminate as soon as all dirt piles are collected. This ensures that the implementation
            # of the TSP agent is equivalent to that of the RL agent
            if 'DirtPiles' in list(factory.state.entities.keys()) and factory.state.entities['DirtPiles'].global_amount == 0.0:
                break
            obs_type, _, _, done, info = factory.step(a)
            if 'DirtPiles' in list(factory.state.entities.keys()):
                still_existing_dirt_piles[-1].append(len(factory.state.entities['DirtPiles']))
            if 'Destinations' in list(factory.state.entities.keys()):
                reached_flags[-1].append(sum([1 for ele in [x.was_reached() for x in factory.state['Destinations']] if ele]))
            ep_steps += 1
            if render:
                factory.render()
            if done:
                break

        cleaned_dirt_piles_per_step = []
        if 'DirtPiles' in list(factory.state.entities.keys()):
            for ep in still_existing_dirt_piles:
                cleaned_dirt_piles_per_step.append([max(ep)-ep[idx] for idx, value in enumerate(ep)])
            # Remove first element and add last element where all dirt piles have been collected
            del cleaned_dirt_piles_per_step[-1][0]
            cleaned_dirt_piles_per_step[-1].append(max(still_existing_dirt_piles[-1]))

        # Add last entry to reached_flags
        print(ep_steps)
        print(reached_flags)
        print(cleaned_dirt_piles_per_step)

        if log:
            if 'DirtPiles' in list(factory.state.entities.keys()):
                metrics_data = {"cleaned_dirt_piles_per_step": cleaned_dirt_piles_per_step}
            if 'Destinations' in list(factory.state.entities.keys()):
                metrics_data = {"reached_flags": reached_flags}
            with open(f"{results_path}/metrics", "wb") as pickle_file:
                pickle.dump(metrics_data, pickle_file)