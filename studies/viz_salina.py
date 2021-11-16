from salina.agents import Agents, TemporalAgent
import torch
from salina import Workspace, get_arguments, get_class, instantiate_class
from pathlib import Path
from salina.agents.gyma import GymAgent
import time
from algorithms.utils import load_yaml_file, add_env_props

if __name__ == '__main__':
    # Setup workspace
    uid = time.time()
    workspace = Workspace()
    weights = Path('/Users/romue/PycharmProjects/EDYS/studies/agent_1636994369.145843.pt')

    cfg = load_yaml_file(Path(__file__).parent / 'sat_mad.yaml')
    add_env_props(cfg)
    cfg['env'].update({'n_agents': 2})

    # instantiate agent and env
    env_agent = GymAgent(
        get_class(cfg['env']),
        get_arguments(cfg['env']),
        n_envs=1
    )

    agents = []
    for _ in range(2):
        a2c_agent = instantiate_class(cfg['agent'])
        if weights:
            a2c_agent.load_state_dict(torch.load(weights))
        agents.append(a2c_agent)

    # combine agents
    acquisition_agent = TemporalAgent(Agents(env_agent, *agents))
    acquisition_agent.seed(42)

    acquisition_agent(workspace, t=0, n_steps=400, stochastic=False, save_render=True)


