from environments.factory.factory_dirt import DirtFactory
from environments.factory.factory_item import ItemFactory


class DirtItemFactory(ItemFactory, DirtFactory):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
