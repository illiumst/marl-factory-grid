How to modify the environment or write modules
===============================================

Modifying levels
----------------
Varying levels are created by defining Walls, Floor or Doors in *.txt*-files (see [levels](marl_factory_grid/levels) for examples).
Define which *level* to use in your *config file* as:

>>> General:
    level_name: rooms  # 'double', 'large', 'simple', ...

... or create your own , maybe with the help of `asciiflow.com <https://asciiflow.com/#/>`_.
Make sure to use `#` as `Walls`_ , `-` as free (walkable) floor, `D` for `Doors`_.
Other Entities (define your own) may bring their own `Symbols`.

.. _Walls: marl_factory_grid/environment/entity/wall.py
.. _Doors: modules/doors/entities.py


Modifying Entites
----------------
Entites are `Objects`_ that can additionally be assigned a position.
Abstract Entities are provided.
If you wish to introduce new entities to the enviroment just create a new module, ...

.. _Objects: marl_factory_grid/environment/entity/object.py

Modifying Groups
----------------
`Groups`_ are entity Sets that provide administrative access to all group members.
All `Entities`_ are available at runtime as EnvState property.

.. _Groups: marl_factory_grid/environment/groups/objects.py
.. _Entities: marl_factory_grid/environment/entity/global_entities.py

Modifying Rules
----------------
`Rules`_ define how the environment behaves on microscale.
Each of the hookes (`on_init`, `pre_step`, `on_step`, '`post_step`', `on_done`)
provide env-access to implement customn logic, calculate rewards, or gather information.
If you wish to introduce new rules to the environment....

.. _Rules: marl_factory_grid/environment/entity/object.py

.. image:: ./images/Hooks_FIKS.png
   :alt: Hooks Image

Modifying Results
----------------
`Results`_ provide a way to return `rule` evaluations such as rewards and state reports
back to the environment.

.. _Results: marl_factory_grid/utils/results.py

Modifying Assets
----------------
Make sure to bring your own assets for each Entity living in the Gridworld as the `Renderer` relies on it.
PNG-files (transparent background) of square aspect-ratio should do the job, in general.

.. image:: ./marl_factory_grid/environment/assets/wall.png
   :alt: Wall Image
.. image:: ./marl_factory_grid/environment/assets/agent/agent.png
   :alt: Agent Image
