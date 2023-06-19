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
Be sure to use '#' as Walls, '-' as free (walkable) Floor-Tiles, 'D' for Doors.
Custom Entites (define you own) may bring their own "Symbol"

#### Entites
Entites are either [Objects](./environment/entity/object.py) for tracking stats or env. [Entity](./environment/entity/entity.py) which can interact.
Abstract Entities are provided.

#### Groups
[Groups](./environment/groups/objects.py) are entity Sets that provide administrative access to all group members. 
All [Entites](./environment/entity/global_entities.py) are available at runtime as EnvState property.


#### Rules
[Rules](./environment/entity/object.py) define how the environment behaves on micro-scale.
Each of the hookes ('on_init', 'pre-step', 'on_step', 'post_step', 'on_done') 
provide env-access to implement customn logic, calculate rewards, or gather information.




[Results](./environment/entity/object.py) provide a way to return 'rule' evaluations such as rewards and state reports 
back to the environment.
#### Assets
Make sure to bring your own assets for each Entity, that is living in the Gridworld, the 'Renderer' relies on it.
PNG-files (transparent background) of square aspect-ratio should do the job, in general.

<div style="margin:0 auto;">
<img src=".\environment\assets\wall.png"  width="10%"> | <img src=".\environment\assets\agent\agent.png"  width="10%">
</div>




