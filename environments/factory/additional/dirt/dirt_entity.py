from environments.factory.base.objects import Entity


class Dirt(Entity):

    @property
    def amount(self):
        return self._amount

    @property
    def encoding(self):
        # Edit this if you want items to be drawn in the ops differntly
        return self._amount

    def __init__(self, *args, amount=None, **kwargs):
        super(Dirt, self).__init__(*args, **kwargs)
        self._amount = amount

    def set_new_amount(self, amount):
        self._amount = amount
        self._collection.notify_change_to_value(self)

    def summarize_state(self, **kwargs):
        state_dict = super().summarize_state(**kwargs)
        state_dict.update(amount=float(self.amount))
        return state_dict
