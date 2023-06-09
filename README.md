# EDYS

Tackling emergent dysfunctions (EDYs) in cooperation with Fraunhofer-IKS

## Setup
1. Make sure to install `virtualenv` using `pip install virtualenv`  
2. Create a new virtual environment `virtualenv venv`
3. Activate the virtual environment `source venv/bin/activate`
4. Install the required dependencies `pip install -r requirements.txt`
## 

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
    level_name: rooms    
```
... or create your own , maybe witht he help of [asciiflow.com](https://asciiflow.com/#/).

#### Entites
TODO
#### Rules
TODO
 - Results
#### Assets
TODO
