from environments.factory.base.registers import EntityCollection
from environments.factory.additional.dest.dest_util import Constants as c
from environments.factory.additional.dest.dest_enitites import Destination


class Destinations(EntityCollection):

    _accepted_objects = Destination

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_blocking_light = False
        self.can_be_shadowed = False

    def as_array(self):
        self._array[:] = c.FREE_CELL
        # ToDo: Switch to new Style Array Put
        # indices = list(zip(range(len(cls)), *zip(*[x.pos for x in cls])))
        # np.put(cls._array, [np.ravel_multi_index(x, cls._array.shape) for x in indices], cls.encodings)
        for item in self:
            if item.pos != c.NO_POS:
                self._array[0, item.x, item.y] = item.encoding
        return self._array

    def __repr__(self):
        return super(Destinations, self).__repr__()


class ReachedDestinations(Destinations):
    _accepted_objects = Destination

    def __init__(self, *args, **kwargs):
        super(ReachedDestinations, self).__init__(*args, **kwargs)
        self.can_be_shadowed = False
        self.is_blocking_light = False

    def __repr__(self):
        return super(ReachedDestinations, self).__repr__()
