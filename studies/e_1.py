import itertools
import random
from pathlib import Path

import simplejson
from stable_baselines3 import DQN, PPO, A2C

from environments.factory.factory_dirt import DirtProperties, DirtFactory
from environments.factory.factory_item import ItemProperties, ItemFactory

if __name__ == '__main__':
    """
    In this studie, we want to explore the macro behaviour of multi agents which are trained on the same task, 
    but never saw each other in training.
    Those agents learned 
    
    
    We start with training a single policy on a single task (dirt cleanup / item pickup).
    Then multiple agent equipped with the same policy are deployed in the same environment.
    
    There are further distinctions to be made:
    
    1. No Observation - ['no_obs']:
    - Agent do not see each other but their consequences of their combined actions
    - Agents can collide
    
    2. Observation in seperate slice - [['seperate_0'], ['seperate_1'], ['seperate_N']]:
    - Agents see other entitys on a seperate slice
    - This slice has been filled with $0 | 1 | \mathbb{N}(0, 1)$
    -- Depending ob the fill value, agents will react diffently
       -> TODO: Test this! 
    
    3. Observation in level slice - ['in_lvl_obs']:
    - This tells the agent to treat other agents as obstacle. 
    - However, the state space is altered since moving obstacles are not part the original agent observation. 
    - We are out of distribution.
    """


def bundle_model(model_class):
    if model_class.__class__.__name__ in ["PPO", "A2C"]:
        kwargs = dict(ent_coef=0.01)
    elif model_class.__class__.__name__ in ["RegDQN", "DQN", "QRDQN"]:
        kwargs = dict(buffer_size=50000,
                      learning_starts=64,
                      batch_size=64,
                      target_update_interval=5000,
                      exploration_fraction=0.25,
                      exploration_final_eps=0.025
                      )
    return lambda: model_class(kwargs)


if __name__ == '__main__':
    # Define a global studi save path
    study_root_path = Path(Path(__file__).stem) / 'out'

    # TODO: Define Global Env Parameters
    factory_kwargs = {


    }

    # TODO: Define global model parameters


    # TODO: Define parameters for both envs
    dirt_props = DirtProperties()
    item_props = ItemProperties()

    # Bundle both environments with global kwargs and parameters
    env_bundles = [lambda: ('dirt', DirtFactory(factory_kwargs, dirt_properties=dirt_props)),
                   lambda: ('item', ItemFactory(factory_kwargs, item_properties=item_props))]

    # Define parameter versions according with #1,2[1,0,N],3
    observation_modes = ['no_obs', 'seperate_0', 'seperate_1', 'seperate_N', 'in_lvl_obs']

    # Define RL-Models
    model_bundles = [bundle_model(model) for model in [A2C, PPO, DQN]]

    # Zip parameters, parameter versions, Env Classes and models
    combinations = itertools.product(model_bundles, env_bundles)

    # Train starts here ############################################################
    # Build Major Loop
    for model, (env_identifier, env_bundle) in combinations:
        for observation_mode in observation_modes:
            # TODO: Create an identifier, which is unique for every combination and easy to read in filesystem
            identifier = f'{model.name}_{observation_mode}_{env_identifier}'
            # Train each combination per seed
            for seed in range(3):
                # TODO: Output folder
                # TODO: Monitor Init
                # TODO: Env Init
                # TODO: Model Init
                # TODO: Model train
                # TODO: Model save
                pass
            # TODO: Seed Compare Plot
    # Train ends here ############################################################

    # Evaluation starts here #####################################################
    # Iterate Observation Modes
    for observation_mode in observation_modes:
        # TODO: For trained policy in study_root_path / identifier
        for policy_group in (x for x in study_root_path.iterdir() if x.is_dir()):
            # TODO: Pick random seed or iterate over available seeds
            policy_seed = next((y for y in study_root_path.iterdir() if y.is_dir()))
            # TODO: retrieve model class
            # TODO: Load both agents
            models = []
            # TODO: Evaluation Loop for i in range(100) Episodes
            for episode in range(100):
                with next(policy_seed.glob('*.yaml')).open('r') as f:
                    env_kwargs = simplejson.load(f)
                # TODO: Monitor Init
                env = None  # TODO: Init Env
                for step in range(400):
                    random_actions = [[random.randint(0, env.n_actions) for _ in range(len(models))] for _ in range(200)]
                    env_state = env.reset()
                    rew = 0
                    for agent_i_action in random_actions:
                        env_state, step_r, done_bool, info_obj = env.step(agent_i_action)
                        rew += step_r
                        if done_bool:
                            break
                print(f'Factory run {episode} done, reward is:\n    {rew}')
            # TODO: Plotting

    pass
