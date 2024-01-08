# About EDYS

## Tackling emergent dysfunctions (EDYs) in cooperation with Fraunhofer-IKS. 

Collaborating with Fraunhofer-IKS, this project is dedicated to investigating Emergent Dysfunctions (EDYs)
within multi-agent environments.

### Project Objectives:

- Create an environment that provokes emerging dysfunctions.

  - This is achieved by creating a high level of background noise in the domain, where various entities perform diverse tasks,
    resulting in a deliberately chaotic dynamic.
  - The goal is to observe and analyze naturally occurring emergent dysfunctions within  the complexity generated in this dynamic environment.


- Observational Framework:

  - The project introduces an environment that is designed to capture dysfunctions as they naturally occur.
  - The environment allows for continuous monitoring of agent behaviors, actions, and interactions.
  - Tracking emergent dysfunctions in real-time provides valuable data for analysis and understanding.


- Compatibility
  - The Framework allows learning entities from different manufacturers and projects with varying representations
  of actions and observations to interact seamlessly within the environment.


- Placeholders
  
  - One can provide an agent with a placeholder observation that contains no information and offers no meaningful insights. 
  - Later, when the environment expands and introduces additional entities available for observation, these new observations can be provided to the agent.
  - This allows for processes such as retraining on an already initialized policy and fine-tuning to enhance the agent's performance based on the enriched information. 


## Setup
Install this environment using `pip install marl-factory-grid`. For more information refer to ['installation'](docs/source/installation.rst).
Refer to [quickstart](_quickstart) for specific scenarios.

## Usage

The majority of environment objects, including entities, rules, and assets, can be loaded automatically. 
Simply specify the requirements of your environment in a [*yaml*-config file](marl_factory_grid/configs/default_config.yaml).

If you only plan on using the environment without making any modifications, use ``quickstart_use``.
This creates a default config-file and another one that lists all possible options of the environment.
Also, it generates an initial script where an agent is executed in the specified environment.
For further details on utilizing the environment, refer to ['usage'](docs/source/usage.rst).

Existing modules include a variety of functionalities within the environment:
- [Agents](marl_factory_grid/algorithms) implement either static strategies or learning algorithms based on the specific configuration.
- Their action set includes opening [door entities](marl_factory_grid/modules/doors/entitites.py), cleaning
[dirt](marl_factory_grid/modules/clean_up/entitites.py), picking up [items](marl_factory_grid/modules/items/entitites.py) and 
delivering them to designated drop-off locations.
- Agents are equipped with a [battery](marl_factory_grid/modules/batteries/entitites.py) that gradually depletes over time if not charged at a chargepod.
- The [maintainer](marl_factory_grid/modules/maintenance/entities.py) aims to repair [machines](marl_factory_grid/modules/machines/entitites.py) that lose health over time.

## Customization

If you plan on modifying the environment by for example adding entities or rules, use ``quickstart_modify``.
This creates a template module and a script that runs an agent, incorporating the generated module. 
More information on how to modify the levels, entities, groups, rules and assets goto [modifications](docs/source/modifications.rst).

### Levels
Varying levels are created by defining Walls, Floor or Doors in *.txt*-files (see [levels](marl_factory_grid/levels) for examples).
Define which *level* to use in your *configfile* as: 
```yaml
General:
    level_name: rooms  # 'double', 'large', 'simple', ...
```
... or create your own , maybe with the help of [asciiflow.com](https://asciiflow.com/#/).
Make sure to use `#` as [Walls](marl_factory_grid/environment/entity/wall.py), `-` as free (walkable) floor, `D` for [Doors](./modules/doors/entities.py).
Other Entites (define you own) may bring their own `Symbols`

### Entites
Entites are [Objects](marl_factory_grid/environment/entity/object.py) that can additionally be assigned a position.
Abstract Entities are provided.

### Groups
[Groups](marl_factory_grid/environment/groups/objects.py) are entity Sets that provide administrative access to all group members. 
All [Entites](marl_factory_grid/environment/entity/global_entities.py) are available at runtime as EnvState property.


### Rules
[Rules](marl_factory_grid/environment/entity/object.py) define how the environment behaves on microscale.
Each of the hookes (`on_init`, `pre_step`, `on_step`, '`post_step`', `on_done`) 
provide env-access to implement customn logic, calculate rewards, or gather information.

![Hooks](../../images/Hooks_FIKS.png)

[Results](marl_factory_grid/environment/entity/object.py) provide a way to return `rule` evaluations such as rewards and state reports 
back to the environment.
### Assets
Make sure to bring your own assets for each Entity living in the Gridworld as the `Renderer` relies on it.
PNG-files (transparent background) of square aspect-ratio should do the job, in general.

<img src="/marl_factory_grid/environment/assets/wall.png"  width="5%"> 
<!--suppress HtmlUnknownAttribute -->
<html &nbsp&nbsp&nbsp&nbsp html> 
<img src="/marl_factory_grid/environment/assets/agent/agent.png"  width="5%">

