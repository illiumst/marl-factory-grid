# EDYS

Tackling emergent dysfunctions (EDYs) in cooperation with Fraunhofer-IKS

## Setup
Just install this environment by `pip install marl-factory-grid`.

## First Steps


### Quickstart
Most of the env. objects (entites, rules and assets) can be loaded automatically. 
Just define what your environment needs in a *yaml*-configfile like:

<details><summary>Example ConfigFile</summary>    
    General:
    level_name: rooms
    env_seed: 69
    verbose: !!bool False
    pomdp_r: 5
    individual_rewards: !!bool True

    Entities:
        Defaults: {}
        Doors:
            closed_on_init: True
            auto_close_interval: 10
            indicate_area: False
        Destinations: {}

    Agents:
        Wolfgang:
            Actions:
                - Move8
                - Noop
                - DoorUse
                - ItemAction
            Observations:
                - All
                - Placeholder
                - Walls
                - Items
                - Placeholder
                - Doors
                - Doors
        Armin:
            Actions:
                - Move4
                - ItemAction
                - DoorUse
            Observations:
                - Combined:
                    - Agent['Wolfgang']
                    - Walls
                    - Doors
                    - Items
    Rules:
        Defaults: {}
        Collision:
            done_at_collisions: !!bool True
        ItemRespawn:
            spawn_freq: 5
        DoorAutoClose: {}

    Assets:
    - Defaults
    - Items
    - Doors
   </details>

Have a look in [\quickstart](./quickstart) for further configuration examples.

### Make it your own

#### Levels
Varying levels are created by defining Walls, Floor or Doors in *.txt*-files (see [./environment/levels](./environment/levels) for examples).
Define which *level* to use in your *configfile* as: 
```yaml
General:
    level_name: rooms  # 'double', 'large', 'simple', ...
```
... or create your own , maybe with the help of [asciiflow.com](https://asciiflow.com/#/).
Make sure to use `#` as [Walls](marl_factory_grid/environment/entity/wall_floor.py), `-` as free (walkable) [Floor](marl_factory_grid/environment/entity/wall_floor.py)-Tiles, `D` for [Walls](./modules/doors/entities.py).
Other Entites (define you own) may bring their own `Symbols`

#### Entites
Entites, either [Objects](marl_factory_grid/environment/entity/object.py) for tracking stats 
or env. [Entity](marl_factory_grid/environment/entity/entity.py) which can interact.
Abstract Entities are provided.

#### Groups
[Groups](marl_factory_grid/environment/groups/objects.py) are entity Sets that provide administrative access to all group members. 
All [Entites](marl_factory_grid/environment/entity/global_entities.py) are available at runtime as EnvState property.


#### Rules
[Rules](marl_factory_grid/environment/entity/object.py) define how the environment behaves on micro-scale.
Each of the hookes (`on_init`, `pre_step`, `on_step`, '`post_step`', `on_done`) 
provide env-access to implement customn logic, calculate rewards, or gather information.

![Hooks](./images/Hooks_FIKS.png)

[Results](marl_factory_grid/environment/entity/object.py) provide a way to return `rule` evaluations such as rewards and state reports 
back to the environment.
#### Assets
Make sure to bring your own assets for each Entity living in the Gridworl as the `Renderer` relies on it.
PNG-files (transparent background) of square aspect-ratio should do the job, in general.

<img src=".\environment\assets\wall.png"  width="5%"> 
&nbsp&nbsp&nbsp&nbsp 
<img src=".\environment\assets\agent\agent.png"  width="5%">



