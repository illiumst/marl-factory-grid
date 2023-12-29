Creating a New Scenario
=======================


Creating a new scenario in the `marl-factory-grid` environment allows you to customize the environment to fit your specific requirements. This guide provides step-by-step instructions on how to create a new scenario, including defining a configuration file, designing a level, and potentially adding new entities, rules, and assets.

### Step 1: Define Configuration File

1. **Create a Configuration File:** Start by creating a new configuration file (`.yaml`) for your scenario. This file will contain settings such as the number of agents, environment dimensions, and other parameters. You can use existing configuration files as templates.

2. **Specify Custom Parameters:** Modify the configuration file to include any custom parameters specific to your scenario. For example, you can set the respawn rate of entities or define specific rewards.

### Step 2: Design the Level

1. **Create a Level File:** Design the layout of your environment by creating a new level file (`.txt`). Use symbols such as `#` for walls, `-` for walkable floors, and introduce new symbols for custom entities (e.g., `A` for apples).

2. **Define Entity Locations:** Specify the initial locations of entities, including agents and any new entities introduced in your scenario. These spawn locations are typically provided in the conf file.

### Step 3: Introduce New Entities

1. **Create New Entity Modules:** If your scenario involves introducing new entities (e.g., apples), create new entity modules in the `marl_factory_grid/environment/entity` directory. Define their behavior, properties, and any custom actions they can perform. Check out the template module.

2. **Update Configuration:** Update the configuration file to include settings related to your new entities, such as spawn rates, initial quantities, or any specific behaviors.

### Step 4: Implement Custom Rules

1. **Create Rule Modules:** If your scenario requires custom rules, create new rule modules in the `marl_factory_grid/environment/rules` directory. Implement the necessary logic to govern the behavior of entities in your scenario and use the provided environment hooks.

2. **Update Configuration:** If your custom rules have configurable parameters, update the configuration file to include these settings and activate the rule by adding it to the conf file.

### Step 5: Add Custom Assets (Optional)

1. **Include Custom Asset Files:** If your scenario introduces new assets (e.g., images for entities), include the necessary asset files in the appropriate directories, such as `marl_factory_grid/environment/assets`.

### Step 6: Test and Experiment

1. **Run Your Scenario:** Use the provided scripts or write your own script to run the scenario with your customized configuration. Observe the behavior of agents and entities in the environment.

2. **Iterate and Experiment:** Adjust configuration parameters, level design, or introduce new elements based on your observations. Iterate through this process until your scenario meets your desired specifications.



Congratulations! You have successfully created a new scenario in the `marl-factory-grid` environment. Experiment with different configurations, levels, entities, and rules to design unique and engaging environments for your simulations.
