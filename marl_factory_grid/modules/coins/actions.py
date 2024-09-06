from typing import Union

from marl_factory_grid.environment.actions import Action
from marl_factory_grid.utils.results import ActionResult

from marl_factory_grid.modules.coins import constants as d

from marl_factory_grid.environment import constants as c


class Collect(Action):

    def __init__(self):
        """
        Attempts to reduce coin amount on entity's position. Fails if no coin is found at the at agents' position.
        """
        super().__init__(d.COLLECT, d.REWARD_COLLECT_VALID, d.REWARD_COLLECT_FAIL)

    def do(self, entity, state) -> Union[None, ActionResult]:
        if coin_pile := next((x for x in state.entities.pos_dict[entity.pos] if "coin" in x.name.lower()), None):
            new_coin_pile_amount = coin_pile.amount - state[d.COIN].collect_amount

            if new_coin_pile_amount <= 0:
                state[d.COIN].delete_env_object(coin_pile)
            else:
                coin_pile.set_new_amount(max(new_coin_pile_amount, c.VALUE_FREE_CELL))
            valid = c.VALID
            print_str = f'{entity.name} did just collect some coins at {entity.pos}.'
            state.print(print_str)

        else:
            valid = c.NOT_VALID
            print_str = f'{entity.name} just tried to collect some coins at {entity.pos}, but failed.'
            state.print(print_str)

        return self.get_result(valid, entity)
