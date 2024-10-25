from marl_factory_grid.algorithms.marl.RL_runner import rerun_coin_quadrant_agent1_training, \
    rerun_two_rooms_agent1_training, rerun_two_rooms_agent2_training, coin_quadrant_multi_agent_rl_eval, \
    two_rooms_multi_agent_rl_eval
from marl_factory_grid.algorithms.static.TSP_runner import coin_quadrant_multi_agent_tsp_eval, \
    two_rooms_multi_agent_tsp_eval


###### Coin-quadrant environment ######
def coin_quadrant_single_agent_training():
    """ Rerun training of RL-agent in coins_quadrant environment.
    The trained model and additional training metrics are saved in the study_out folder. """
    rerun_coin_quadrant_agent1_training()


def coin_quadrant_RL_multi_agent_eval_emergent():
    """ Rerun multi-agent evaluation of RL-agents in coins_quadrant environment,
    with occurring emergent phenomenon. Evaluation takes trained models from study_out/run0 for both agents."""
    coin_quadrant_multi_agent_rl_eval(emergent_phenomenon=True)


def coin_quadrant_RL_multi_agent_eval_prevented():
    """ Rerun multi-agent evaluation of RL-agents in coins_quadrant environment,
    with emergence prevention mechanism. Evaluation takes trained models from study_out/run0 for both agents."""
    coin_quadrant_multi_agent_rl_eval(emergent_phenomenon=False)


def coin_quadrant_TSP_multi_agent_eval_emergent():
    """ Rerun multi-agent evaluation of TSP-agents in coins_quadrant environment,
    with occurring emergent phenomenon. """
    coin_quadrant_multi_agent_tsp_eval(emergent_phenomenon=True)


def coin_quadrant_TSP_multi_agent_eval_prevented():
    """ Rerun multi-agent evaluation of TSP-agents in coins_quadrant environment,
    with emergence prevention mechanism. """
    coin_quadrant_multi_agent_tsp_eval(emergent_phenomenon=False)


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
    coin_quadrant_RL_multi_agent_eval_prevented()
