import pickle

import numpy as np
from scipy import stats

import wandb

from marl_factory_grid.algorithms.rl.RL_runner import rerun_dirt_quadrant_agent1_training, \
    rerun_two_rooms_agent1_training, rerun_two_rooms_agent2_training, dirt_quadrant_multi_agent_rl_eval, \
    two_rooms_multi_agent_rl_eval, single_agent_eval
from marl_factory_grid.algorithms.tsp.TSP_runner import dirt_quadrant_multi_agent_tsp_eval, \
    two_rooms_multi_agent_tsp_eval
from marl_factory_grid.utils.plotting.plot_single_runs import plot_reached_flags_per_step, \
    plot_collected_coins_per_step, plot_performance_distribution_on_coin_quadrant, \
    plot_reached_flags_per_step_with_error


###### Coin-quadrant environment ######
def coin_quadrant_single_agent_training():
    """ Rerun training of RL-agent in coins_quadrant (dirt_quadrant) environment.
        The trained model and additional training metrics are saved in the study_out folder. """
    rerun_dirt_quadrant_agent1_training()


def coin_quadrant_RL_multi_agent_eval_emergent():
    """ Rerun multi-agent evaluation of RL-agents in coins_quadrant (dirt_quadrant)
        environment, with occurring emergent phenomenon. Evaluation takes trained models
        from study_out/run0 for both agents."""
    dirt_quadrant_multi_agent_rl_eval(emergent_phenomenon=True)


def coin_quadrant_RL_multi_agent_eval_prevented():
    """ Rerun multi-agent evaluation of RL-agents in coins_quadrant (dirt_quadrant)
        environment, with emergence prevention mechanism. Evaluation takes trained models
        from study_out/run0 for both agents."""
    dirt_quadrant_multi_agent_rl_eval(emergent_phenomenon=False)


def coin_quadrant_TSP_multi_agent_eval_emergent():
    """ Rerun multi-agent evaluation of TSP-agents in coins_quadrant (dirt_quadrant)
        environment, with occurring emergent phenomenon. """
    dirt_quadrant_multi_agent_tsp_eval(emergent_phenomenon=True)


def coin_quadrant_TSP_multi_agent_eval_prevented():
    """ Rerun multi-agent evaluation of TSP-agents in coins_quadrant (dirt_quadrant)
        environment, with emergence prevention mechanism. """
    dirt_quadrant_multi_agent_tsp_eval(emergent_phenomenon=False)


###### Two-rooms environment ######

def two_rooms_agent1_training():
    """ Rerun training of left RL-agent in two_rooms environment.
        The trained model and additional training metrics are saved in the study_out folder. """
    rerun_two_rooms_agent1_training()


def two_rooms_agent2_training():
    """ Rerun training of right RL-agent in two_rooms environment.
        The trained model and additional training metrics are saved in the study_out folder. """
    rerun_two_rooms_agent2_training()


def two_rooms_RL_multi_agent_eval_emergent():
    """ Rerun multi-agent evaluation of RL-agents in two_rooms environment, with
        occurring emergent phenomenon. Evaluation takes trained models
        from study_out/run1 for agent1 and study_out/run2 for agent2. """
    two_rooms_multi_agent_rl_eval(emergent_phenomenon=True)


def two_rooms_RL_multi_agent_eval_prevented():
    """ Rerun multi-agent evaluation of RL-agents in two_rooms environment, with
        emergence prevention mechanism. Evaluation takes trained models
        from study_out/run1 for agent1 and study_out/run2 for agent2. """
    two_rooms_multi_agent_rl_eval(emergent_phenomenon=False)


def two_rooms_TSP_multi_agent_eval_emergent():
    """ Rerun multi-agent evaluation of TSP-agents in two_rooms environment, with
        occurring emergent phenomenon. """
    two_rooms_multi_agent_tsp_eval(emergent_phenomenon=True)


def two_rooms_TSP_multi_agent_eval_prevented():
    """ Rerun multi-agent evaluation of TSP-agents in two_rooms environment, with
        emergence prevention mechanism. """
    two_rooms_multi_agent_tsp_eval(emergent_phenomenon=False)


if __name__ == '__main__':
    # Select any of the above functions to rerun the respective part
    #  from our evaluation section of the paper
    #coin_quadrant_RL_multi_agent_eval_prevented()
    two_rooms_RL_multi_agent_eval_prevented()
    #coin_quadrant_TSP_multi_agent_eval_prevented()
    #two_rooms_RL_multi_agent_eval_prevented()
    #plot_reached_flags_per_step("/Users/julian/Coding/Projects/PyCharmProjects/EDYS/study_out/run36/metrics", "/Users/julian/Coding/Projects/PyCharmProjects/EDYS/study_out/tsp_run0/metrics", "/Users/julian/Coding/Projects/PyCharmProjects/EDYS/study_out")
    #plot_reached_flags_per_step(["run36", "run37"], ["tsp_run0", "tsp_run1"], "/Users/julian/Coding/Projects/PyCharmProjects/EDYS/study_out")
    #plot_collected_coins_per_step(["run38", "run39"], ["tsp_run2", "tsp_run3"], "/Users/julian/Coding/Projects/PyCharmProjects/EDYS/study_out")
    """return_runs = []
    for i in range(11, 21):
        with open(f"/Users/julian/Coding/Projects/PyCharmProjects/EDYS/study_out/run{i}/metrics", "rb") as pickle_file:
            metrics = pickle.load(pickle_file)
        return_runs.append(metrics["return_development"])
    mean_return_runs = []"""
    dirt_quadrant = {"RL_emergence": [20, 13, 13, 20, 19, 16, 19, 20, 20, 19, 19, 19, 13, 18, 18, 20, 20, 12, 20, 13],
                     "RL_prevented": [14.555555555555555, 14.555555555555555, 14.555555555555555, 11.444444444444445, 19.0,
                                      16.0, 14.555555555555555, 13.0, 14.555555555555555, 13.0, 19.0, 19.0, 14.555555555555555,
                                      18.0, 18.0, 13.0, 10.0, 13.777777777777779, 13.0, 13.0],
                     "TSP_emergence": [20, 13, 13, 20, 13, 16, 19, 20, 20, 19, 19, 19, 15, 18, 18, 20, 10, 12, 20, 13],
                     "TSP_prevented": [13, 13, 13, 9, 13, 11, 13, 13, 13, 13, 13, 15, 12, 12, 12, 13, 10, 12, 13, 13]}

    #plot_performance_distribution_on_coin_quadrant(dirt_quadrant, results_path="/Users/julian/Coding/Projects/PyCharmProjects/EDYS/study_out", grid=True)

    """plot_reached_flags_per_step_with_error(mean_steps_RL_prevented=[0, 12.0, 13.0], error_bars_RL_prevented=[0, 0.0, 0.0],
                                           mean_steps_TSP_prevented=[0, 15.2, 16.8], error_bars_TSP_prevented=[0, 0.45243143257081975, 0.7388174355966146],
                                           flags_reached=[0, 1, 2], results_path="/Users/julian/Coding/Projects/PyCharmProjects/EDYS/study_out", grid=False)"""