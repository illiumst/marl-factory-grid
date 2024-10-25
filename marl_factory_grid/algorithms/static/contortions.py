import numpy as np
from marl_factory_grid.algorithms.static.TSP_coin_agent import TSPCoinAgent
from marl_factory_grid.algorithms.static.TSP_target_agent import TSPTargetAgent


def get_coin_quadrant_tsp_agents(emergent_phenomenon, factory):
    agents = [TSPCoinAgent(factory, 0), TSPCoinAgent(factory, 1)]
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