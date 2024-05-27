import os
from pathlib import Path

import numpy as np
from tqdm import trange

from marl_factory_grid.algorithms.tsp.TSP_dirt_agent import TSPDirtAgent
from marl_factory_grid.algorithms.tsp.TSP_target_agent import TSPTargetAgent
from marl_factory_grid.environment.factory import Factory


def get_dirt_quadrant_tsp_agents(emergent_phenomenon, factory):
    agents = [TSPDirtAgent(factory, 0), TSPDirtAgent(factory, 1)]
    if not emergent_phenomenon:
        edge_costs = {}
        # Add costs for horizontal edges
        for i in range(1, 10):
            for j in range(1, 9):
                # Add costs for both traversal directions
                edge_costs[f"{(i, j)}-{i, j + 1}"] = 0.55 + (i - 1) * 0.05
                edge_costs[f"{i, j + 1}-{(i, j)}"] = 0.55 + (i - 1) * 0.05

        # Add costs for vertical edges
        for i in range(1, 9):
            for j in range(1, 10):
                # Add costs for both traversal directions
                edge_costs[f"{(i, j)}-{i + 1, j}"] = 0.55 + (i) * 0.05
                edge_costs[f"{i + 1, j}-{(i, j)}"] = 0.55 + (i - 1) * 0.05


        for agent in agents:
            for u, v, weight in agent._position_graph.edges(data='weight'):
                agent._position_graph[u][v]['weight'] = edge_costs[f"{u}-{v}"]


    return agents


def get_two_rooms_tsp_agents(emergent_phenomenon, factory):
    agents = [TSPTargetAgent(factory, 0), TSPTargetAgent(factory, 1)]
    if not emergent_phenomenon:
        edge_costs = {}
        # Add costs for horizontal edges
        for i in range(1, 6):
            for j in range(1, 13):
                # Add costs for both traversal directions
                edge_costs[f"{(i, j)}-{i, j + 1}"] = np.abs(5/i*np.cbrt(((j+1)/4 - 1)) - 1)
                edge_costs[f"{i, j + 1}-{(i, j)}"] = np.abs(5/i*np.cbrt((j/4 - 1)) - 1)

        # Add costs for vertical edges
        for i in range(1, 5):
            for j in range(1, 14):
                # Add costs for both traversal directions
                edge_costs[f"{(i, j)}-{i + 1, j}"] = np.abs(5/(i+1)*np.cbrt((j/4 - 1)) - 1)
                edge_costs[f"{i + 1, j}-{(i, j)}"] = np.abs(5/i*np.cbrt((j/4 - 1)) - 1)


        for agent in agents:
            for u, v, weight in agent._position_graph.edges(data='weight'):
                agent._position_graph[u][v]['weight'] = edge_costs[f"{u}-{v}"]
    return agents


def run_tsp_setting(config_name, emergent_phenomenon):
    # Render at each step?
    render = True

    # Path to config File
    path = Path(f'../marl_factory_grid/environment/configs/tsp/{config_name}.yaml')

    # Create results folder
    runs = os.listdir("../study_out/")
    run_numbers = [int(run[7:]) for run in runs if run[:7] == "tsp_run"]
    next_run_number = max(run_numbers) + 1 if run_numbers else 0
    results_path = f"../study_out/tsp_run{next_run_number}"
    os.mkdir(results_path)

    # Env Init
    factory = Factory(path)

    with open(f"{results_path}/env_config.txt", "w") as txt_file:
        txt_file.write(str(factory.conf))

    for episode in trange(1):
        _ = factory.reset()
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
        while not done:
            a = [x.predict() for x in agents]
            # Have this condition, to terminate as soon as all dirt piles are collected. This ensures that the implementation
            # of the TSP agent is equivalent to that of the RL agent
            if 'DirtPiles' in list(factory.state.entities.keys()) and factory.state.entities['DirtPiles'].global_amount == 0.0:
                break
            obs_type, _, _, done, info = factory.step(a)
            if render:
                factory.render()
            if done:
                break


def dirt_quadrant_multi_agent_tsp(emergent_phenomenon):
    run_tsp_setting("dirt_quadrant", emergent_phenomenon)


def two_rooms_multi_agent_tsp(emergent_phenomenon):
    run_tsp_setting("two_rooms", emergent_phenomenon)


if __name__ == '__main__':
    dirt_quadrant_multi_agent_tsp(False)
