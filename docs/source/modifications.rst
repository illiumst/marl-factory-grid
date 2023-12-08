How to modify the environment or write modules
===============================================

Modifying levels
----------------
Varying levels are created by defining Walls, Floor or Doors in *.txt*-files (see `levels`_ for examples).
Define which *level* to use in your *config file* as:

.. _levels: marl_factory_grid/levels

>>> General:
    level_name: rooms  # 'simple', 'narrow_corridor', 'eight_puzzle',...

... or create your own. Maybe with the help of `asciiflow.com <https://asciiflow.com/#/>`_.
Make sure to use `#` as `Walls`_ , `-` as free (walkable) floor and `D` for `Doors`_.
Other Entities (define your own) may bring their own `Symbols`.

.. _Walls: marl_factory_grid/environment/entity/wall.py
.. _Doors: modules/doors/entities.py


Modifying Entites
----------------
Entities are `Objects`_ that can additionally be assigned a position.
Abstract Entities are provided.

If you wish to introduce new entities to the environment just create a new module that implements the entity class. If
necessary, provide additional classe such as custom actions or rewards and load the entity into the environment using
the config file.

.. _Objects: marl_factory_grid/environment/entity/object.py

Modifying Groups
----------------
`Groups`_ are entity Sets that provide administrative access to all group members.
All `Entity Collections`_ are available at runtime as a property of the env state.
If you add an entity, you probably also want a collection of that entity.

.. _Groups: marl_factory_grid/environment/groups/objects.py
.. _Entity Collections: marl_factory_grid/environment/entity/global_entities.py

Modifying Rules
----------------
`Rules`_ define how the environment behaves on micro scale.
Each of the hooks (`on_init`, `pre_step`, `on_step`, '`post_step`', `on_done`) provide env-access to implement custom
logic, calculate rewards, or gather information.

If you wish to introduce new rules to the environment make sure it implements the Rule class and override its' hooks
to implement your own rule logic.

.. _Rules: marl_factory_grid/environment/entity/object.py

.. image:: ./images/Hooks_FIKS.png
   :alt: Hooks Image

Modifying Results
----------------
`Results`_ provide a way to return `rule` evaluations such as rewards and state reports back to the environment.

.. _Results: marl_factory_grid/utils/results.py

Modifying Assets
----------------
Make sure to bring your own assets for each Entity living in the Gridworld as the `Renderer` relies on it.
PNG-files (transparent background) of square aspect-ratio should do the job, in general.

.. image:: ./marl_factory_grid/environment/assets/wall.png
   :alt: Wall Image
.. image:: ./marl_factory_grid/environment/assets/agent/agent.png
   :alt: Agent Image
