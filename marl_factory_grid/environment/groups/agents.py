from marl_factory_grid.environment.entity.agent import Agent
from marl_factory_grid.environment.groups.collection import Collection


class Agents(Collection):
    _entity = Agent

    @property
    def obs_pairs(self):
        pair_list = [(self.name, self)]
        pair_list.extend([(a.name, a) for a in self])
        return pair_list

    @property
    def spawn_rule(self):
        return {}

    @property
    def var_is_blocking_light(self):
        return False

    @property
    def var_can_move(self):
        return True

    @property
    def var_has_position(self):
        return True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def action_space(self):
        """
        TODO


        :return:
        """
        from gymnasium import spaces
        space = spaces.Tuple([spaces.Discrete(len(x.actions)) for x in self])
        return space

    @property
    def named_action_space(self) -> dict[str, dict[str, list[int]]]:
        """
        TODO


        :return:
        """
        named_space = dict()
        for agent in self:
            named_space[agent.name] = {action.name: idx for idx, action in enumerate(agent.actions)}
        return named_space
