# EDYS

Tackling emergent dysfunctions (EDYs) in cooperation with Fraunhofer-IKS

## Setup
Install this environment using `pip install marl-factory-grid`.

## First Steps


### Quickstart
Most of the env. objects (entites, rules and assets) can be loaded automatically. 
Just define what your environment needs in a *yaml*-configfile like:

<details><summary>Example ConfigFile</summary>    
    
    # Default Configuration File
    
    General:
      # RNG-seed to sample the same "random" numbers every time, to make the different runs comparable.
      env_seed: 69
      # Individual vs global rewards
      individual_rewards: true
      # The level.txt file to load from marl_factory_grid/levels
      level_name: large
      # View Radius; 0 = full observatbility
      pomdp_r: 3
      # Print all messages and events
      verbose: false
      # Run tests
      tests: false
    
    # Agents section defines the characteristics of different agents in the environment.
    
    # An Agent requires a list of actions and observations.
    # Possible actions: Noop, Charge, Clean, DestAction, DoorUse, ItemAction, MachineAction, Move8, Move4, North, NorthEast, ...
    # Possible observations: All, Combined, GlobalPosition, Battery, ChargePods, DirtPiles, Destinations, Doors, Items, Inventory, DropOffLocations, Maintainers, ...
    # You can use 'clone' as the agent name to have multiple instances with either a list of names or an int specifying the number of clones.
    Agents:
      Wolfgang:
        Actions:
          - Noop
          - Charge
          - Clean
          - DestAction
          - DoorUse
          - ItemAction
          - Move8
        Observations:
          - Combined:
              - Other
              - Walls
          - GlobalPosition
          - Battery
          - ChargePods
          - DirtPiles
          - Destinations
          - Doors
          - Items
          - Inventory
          - DropOffLocations
          - Maintainers
    
    # Entities section defines the initial parameters and behaviors of different entities in the environment.
    # Entities all spawn using coords_or_quantity, a number of entities or coordinates to place them.
    Entities:
      # Batteries: Entities representing power sources for agents.
      Batteries:
        initial_charge: 0.8
        per_action_costs: 0.02
    
      # ChargePods: Entities representing charging stations for Batteries.
      ChargePods:
        coords_or_quantity: 2
    
      # Destinations: Entities representing target locations for agents.
      # - spawn_mode: GROUPED or SINGLE. Determines how destinations are spawned.
      Destinations:
        coords_or_quantity: 1
        spawn_mode: GROUPED
    
      # DirtPiles: Entities representing piles of dirt.
      # - initial_amount: Initial amount of dirt in each pile.
      # - clean_amount: Amount of dirt cleaned in each cleaning action.
      # - dirt_spawn_r_var: Random variation in dirt spawn amounts.
      # - max_global_amount: Maximum total amount of dirt allowed in the environment.
      # - max_local_amount: Maximum amount of dirt allowed in one position.
      DirtPiles:
        coords_or_quantity: 10
        initial_amount: 2
        clean_amount: 1
        dirt_spawn_r_var: 0.1
        max_global_amount: 20
        max_local_amount: 5
    
      # Doors are spawned using the level map.
      Doors:
    
      # DropOffLocations: Entities representing locations where agents can drop off items.
      # - max_dropoff_storage_size: Maximum storage capacity at each drop-off location.
      DropOffLocations:
        coords_or_quantity: 1
        max_dropoff_storage_size: 0
    
      # GlobalPositions.
      GlobalPositions: { }
    
      # Inventories: Entities representing inventories for agents.
      Inventories: { }
    
      # Items: Entities representing items in the environment.
      Items:
        coords_or_quantity: 5
    
      # Machines: Entities representing machines in the environment.
      Machines:
        coords_or_quantity: 2
    
      # Maintainers: Entities representing maintainers that aim to maintain machines.
      Maintainers:
        coords_or_quantity: 1
    
    
    # Rules section specifies the rules governing the dynamics of the environment.
    Rules:
      # Environment Dynamics
      # When stepping over a dirt pile, entities carry a ratio of the dirt to their next position
      EntitiesSmearDirtOnMove:
        smear_ratio: 0.2
      # Doors automatically close after a certain number of time steps
      DoorAutoClose:
        close_frequency: 10
      # Maintainers move at every time step
      MoveMaintainers:
    
      # Respawn Stuff
      # Define how dirt should respawn after the initial spawn
      RespawnDirt:
        respawn_freq: 15
      # Define how items should respawn after the initial spawn
      RespawnItems:
        respawn_freq: 15
    
      # Utilities
      # This rule defines the collision mechanic, introduces a related DoneCondition and lets you specify rewards.
      # Can be omitted/ignored if you do not want to take care of collisions at all.
      WatchCollisions:
        done_at_collisions: false
    
      # Done Conditions
      # Define the conditions for the environment to stop. Either success or a fail conditions.
      # The environment stops when an agent reaches a destination
      DoneAtDestinationReach:
      # The environment stops when all dirt is cleaned
      DoneOnAllDirtCleaned:
      # The environment stops when a battery is discharged
      DoneAtBatteryDischarge:
      # The environment stops when a maintainer reports a collision
      DoneAtMaintainerCollision:
      # The environment stops after max steps
      DoneAtMaxStepsReached:
        max_steps: 500

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
Make sure to use `#` as [Walls](marl_factory_grid/environment/entity/wall.py), `-` as free (walkable) floor, `D` for [Walls](./modules/doors/entities.py).
Other Entites (define you own) may bring their own `Symbols`

#### Entites
Entites are [Objects](marl_factory_grid/environment/entity/object.py) that can additionally be assigned a position.
Abstract Entities are provided.

#### Groups
[Groups](marl_factory_grid/environment/groups/objects.py) are entity Sets that provide administrative access to all group members. 
All [Entites](marl_factory_grid/environment/entity/global_entities.py) are available at runtime as EnvState property.


#### Rules
[Rules](marl_factory_grid/environment/entity/object.py) define how the environment behaves on microscale.
Each of the hookes (`on_init`, `pre_step`, `on_step`, '`post_step`', `on_done`) 
provide env-access to implement customn logic, calculate rewards, or gather information.

![Hooks](./images/Hooks_FIKS.png)

[Results](marl_factory_grid/environment/entity/object.py) provide a way to return `rule` evaluations such as rewards and state reports 
back to the environment.
#### Assets
Make sure to bring your own assets for each Entity living in the Gridworld as the `Renderer` relies on it.
PNG-files (transparent background) of square aspect-ratio should do the job, in general.

<img src="/marl_factory_grid/environment/assets/wall.png"  width="5%"> 
<!--suppress HtmlUnknownAttribute -->
<html &nbsp&nbsp&nbsp&nbsp html> 
<img src="/marl_factory_grid/environment/assets/agent/agent.png"  width="5%">



