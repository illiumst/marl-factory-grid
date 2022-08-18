from environments.factory.base.objects import Entity


class DirtPile(Entity):

    @property
    def amount(self):
        return self._amount

    @property
    def encoding(self):
        # Edit this if you want items to be drawn in the ops differntly
        return self._amount

    def __init__(self, *args, amount=None, **kwargs):
        super(DirtPile, self).__init__(*args, **kwargs)
        self._amount = amount

    def set_new_amount(self, amount):
        self._amount = amount
        self._collection.notify_change_to_value(self)

    def summarize_state(self):
        state_dict = super().summarize_state()
        state_dict.update(amount=float(self.amount))
        return state_dict
