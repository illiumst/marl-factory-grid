# About EDYS

## Tackling emergent dysfunctions (EDYs) in cooperation with Fraunhofer-IKS.

Collaborating with Fraunhofer-IKS, this project is dedicated to investigating Emergent Dysfunctions (EDYs) within
multi-agent environments. In multi-agent reinforcement learning (MARL), a population of agents learns by interacting
with each other in a shared environment and adapt their behavior based on the feedback they receive from the environment
and the actions of other agents.

In this context, emergent behavior describes spontaneous behaviors resulting from interactions among agents and
environmental stimuli, rather than explicit programming. This promotes natural, adaptable behavior, increases system
unpredictability for dynamic learning , enables diverse strategies, and encourages collective intelligence for complex
problem-solving. However, the complex dynamics of the environment also give rise to emerging dysfunctions—unexpected
issues from agent interactions. This research aims to enhance our understanding of EDYs and their impact on multi-agent
systems.

### Project Objectives:

- Create an environment that provokes emerging dysfunctions.

    - This is achieved by creating a high level of background noise in the domain, where various entities perform
      diverse tasks, resulting in a deliberately chaotic dynamic.
    - The goal is to observe and analyze naturally occurring emergent dysfunctions within the complexity generated in
      this dynamic environment.


- Observational Framework:

    - The project introduces an environment that is designed to capture dysfunctions as they naturally occur.
    - The environment allows for continuous monitoring of agent behaviors, actions, and interactions.
    - Tracking emergent dysfunctions in real-time provides valuable data for analysis and understanding.


- Compatibility
    - The Framework allows learning entities from different manufacturers and projects with varying representations
      of actions and observations to interact seamlessly within the environment.


## Setup

Install this environment using `pip install marl-factory-grid`. For more information refer
to ['installation'](docs/source/installation.rst).

## Usage

The environment is configured to automatically load necessary objects, including entities, rules, and assets, based on your requirements.
You can utilize existing configurations to replicate the experiments from [this paper](PAPER).

- Preconfigured Studies:
    The studies folder contains predefined studies that can be used to replicate the experiments.
    These studies provide a structured way to validate and analyze the outcomes observed in different scenarios.
  - Creating your own scenarios:
      If you want to use the environment with custom entities, rules or levels refer to the [complete repository]().



Existing modules include a variety of functionalities within the environment:

- [Agents](marl_factory_grid/algorithms) implement either static strategies or learning algorithms based on the specific
  configuration.
- Their action set includes opening [door entities](marl_factory_grid/modules/doors/entitites.py), collecting [coins](marl_factory_grid/modules/coins/coin  cleaning
  [dirt](marl_factory_grid/modules/clean_up/entitites.py), picking
  up [items](marl_factory_grid/modules/items/entitites.py) and
  delivering them to designated drop-off locations.
- Agents are equipped with a [battery](marl_factory_grid/modules/batteries/entitites.py) that gradually depletes over
  time if not charged at a chargepod.
- The [maintainer](marl_factory_grid/modules/maintenance/entities.py) aims to
  repair [machines](marl_factory_grid/modules/machines/entitites.py) that lose health over time.


## Limitations

The provided code and documentation are tailored for replicating and validating experiments as described in the paper. 
Modifications to the environment, such as adding new entities, creating additional rules, or customizing behavior beyond the provided scope are not supported in this release.
If you are interested in accessing the complete project, including features not covered in this release, refer to the [full repository](LINK FULL REPO).

For further details on running the experiments, please consult the relevant documentation provided in the studies' folder.
