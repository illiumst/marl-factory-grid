from typing import List, Tuple, Union, Dict

from marl_factory_grid.environment.entity.entity import Entity
from marl_factory_grid.environment.groups.objects import Objects
from marl_factory_grid.environment.entity.object import Object
import marl_factory_grid.environment.constants as c
from marl_factory_grid.utils.results import Result


class Collection(Objects):
    _entity = Object  # entity?
    symbol = None

    @property
    def var_is_blocking_light(self):
        """
       Indicates whether the collection blocks light.

       :return: Always False for a collection.
       """
        return False

    @property
    def var_is_blocking_pos(self):
        """
        Indicates whether the collection blocks positions.

        :return: Always False for a collection.
        """
        return False

    @property
    def var_can_collide(self):
        """
        Indicates whether the collection can collide.

        :return: Always False for a collection.
        """
        return False

    @property
    def var_can_move(self):
        """
        Indicates whether the collection can move.

        :return: Always False for a collection.
        """
        return False

    @property
    def var_has_position(self):
        """
        Indicates whether the collection has positions.

        :return: Always True for a collection.
        """
        return True

    @property
    def encodings(self):
        """
        Returns a list of encodings for all entities in the collection.

        :return: List of encodings.
        """
        return [x.encoding for x in self]

    @property
    def spawn_rule(self):
        """
        Prevents SpawnRule creation if Objects are spawned by the map, doors, etc.

        :return: The spawn rule or None.
        """
        if self.symbol:
            return None
        elif self._spawnrule:
            return self._spawnrule
        else:
            return {c.SPAWN_ENTITY_RULE: dict(collection=self, coords_or_quantity=self._coords_or_quantity)}

    def __init__(self, size, *args, coords_or_quantity: int = None, ignore_blocking=False,
                 spawnrule: Union[None, Dict[str, dict]] = None,
                 **kwargs):
        """
        Initializes the Collection.

        :param size: Size of the collection.
        :type size: int
        :param coords_or_quantity: Coordinates or quantity for spawning entities.
        :param ignore_blocking: Ignore blocking when spawning entities.
        :type ignore_blocking: bool
        :param spawnrule: Spawn rule for the collection. Default: None
        :type spawnrule:  Union[None, Dict[str, dict]]
        """
        super(Collection, self).__init__(*args, **kwargs)
        self._coords_or_quantity = coords_or_quantity
        self.size = size
        self._spawnrule = spawnrule
        self._ignore_blocking = ignore_blocking

    def trigger_spawn(self, state, *entity_args, coords_or_quantity=None, ignore_blocking=False,  **entity_kwargs):
        """
        Triggers the spawning of entities in the collection.

        :param state: The game state.
        :type state: marl_factory_grid.utils.states.GameState
        :param entity_args: Additional arguments for entity creation.
        :param coords_or_quantity: Coordinates or quantity for spawning entities.
        :param ignore_blocking: Ignore blocking when spawning entities.
        :param entity_kwargs: Additional keyword arguments for entity creation.
        :return: Result of the spawn operation.
        """
        coords_or_quantity = coords_or_quantity if coords_or_quantity else self._coords_or_quantity
        if self.var_has_position:
            if self.var_has_position and isinstance(coords_or_quantity, int):
                if ignore_blocking or self._ignore_blocking:
                    coords_or_quantity = state.entities.floorlist[:coords_or_quantity]
                else:
                    coords_or_quantity = state.get_n_random_free_positions(coords_or_quantity)
            self.spawn(coords_or_quantity, *entity_args,  **entity_kwargs)
            state.print(f'{len(coords_or_quantity)} new {self.name} have been spawned at {coords_or_quantity}')
            return Result(identifier=f'{self.name}_spawn', validity=c.VALID, value=len(coords_or_quantity))
        else:
            if isinstance(coords_or_quantity, int):
                self.spawn(coords_or_quantity, *entity_args,  **entity_kwargs)
                state.print(f'{coords_or_quantity} new {self.name} have been spawned randomly.')
                return Result(identifier=f'{self.name}_spawn', validity=c.VALID, value=coords_or_quantity)
            else:
                raise ValueError(f'{self._entity.__name__} has no position!')

    def spawn(self, coords_or_quantity: Union[int, List[Tuple[(int, int)]]], *entity_args, **entity_kwargs):
        """
        Spawns entities in the collection.

        :param coords_or_quantity: Coordinates or quantity for spawning entities.
        :param entity_args: Additional arguments for entity creation.
        :param entity_kwargs: Additional keyword arguments for entity creation.
        :return: Validity of the spawn operation.
        """
        if self.var_has_position:
            if isinstance(coords_or_quantity, int):
                raise ValueError(f'{self._entity.__name__} should have a position!')
            else:
                self.add_items([self._entity(pos, *entity_args, **entity_kwargs) for pos in coords_or_quantity])
        else:
            if isinstance(coords_or_quantity, int):
                self.add_items([self._entity(*entity_args, **entity_kwargs) for _ in range(coords_or_quantity)])
            else:
                raise ValueError(f'{self._entity.__name__} has no  position!')
        return c.VALID

    def despawn(self, items: List[Object]):
        """
        Despawns entities from the collection.

        :param items: List of entities to despawn.
        """
        items = [items] if isinstance(items, Object) else items
        for item in items:
            del self[item]

    def add_item(self, item: Entity):
        assert self.var_has_position or (len(self) <= self.size)
        super(Collection, self).add_item(item)
        return self

    def delete_env_object(self, env_object):
        """
        Deletes an environmental object from the collection.

        :param env_object: The environmental object to delete.
        """
        del self[env_object.name]

    def delete_env_object_by_name(self, name):
        """
        Deletes an environmental object from the collection by name.

        :param name: The name of the environmental object to delete.
        """
        del self[name]

    @property
    def obs_pairs(self):
        pair_list = [(self.name, self)]
        try:
            if self.var_can_be_bound:
                pair_list.extend([(a.name, a) for a in self])
        except AttributeError:
            pass
        return pair_list

    def by_entity(self, entity):
        try:
            return next((x for x in self if x.belongs_to_entity(entity)))
        except (StopIteration, AttributeError):
            return None

    def render(self):
        if self.var_has_position:
            return [y for y in [x.render() for x in self] if y is not None]
        else:
            return []

    @classmethod
    def from_coordinates(cls, positions: [(int, int)], *args, entity_kwargs=None, **kwargs, ):
        """
       Creates a collection of entities from specified coordinates.

       :param positions: List of coordinates for entity positions.
       :param args: Additional positional arguments.
       :return: The created collection.
       """
        collection = cls(*args, **kwargs)
        collection.add_items(
            [cls._entity(tuple(pos), **entity_kwargs if entity_kwargs is not None else {}) for pos in positions])
        return collection

    def __delitem__(self, name):
        idx, obj = next((i, obj) for i, obj in enumerate(self) if obj.name == name)
        try:
            for observer in obj.observers:
                observer.notify_del_entity(obj)
        except AttributeError:
            pass
        super().__delitem__(name)

    def by_pos(self, pos: (int, int)):
        """
        Retrieves an entity from the collection based on its position.

        :param pos: The position tuple.
        :return: The entity at the specified position or None if not found.
        """
        pos = tuple(pos)
        try:
            return self.pos_dict[pos]
        except StopIteration:
            pass
        except ValueError:
            pass

    @property
    def positions(self):
        """
        Returns a list of positions for all entities in the collection.

        :return: List of positions.
        """
        return [e.pos for e in self]

    def notify_del_entity(self, entity: Entity):
        try:
            self.pos_dict[entity.pos].remove(entity)
        except (ValueError, AttributeError):
            pass
