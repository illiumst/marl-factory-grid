import unittest
from typing import List

import marl_factory_grid.modules.maintenance.constants as M
from marl_factory_grid.modules import Door, Machine
from marl_factory_grid.utils.results import TickResult, DoneResult, ActionResult
import marl_factory_grid.environment.constants as c


class Test(unittest.TestCase):

    @property
    def name(self):
        return self.__class__.__name__

    def __init__(self):
        """
        Base test class for unit tests.
        """
        super().__init__()

    def __repr__(self):
        return f'{self.name}'

    def on_init(self, state, lvl_map):
        return []

    def on_reset(self):
        return []

    def tick_pre_step(self, state) -> List[TickResult]:
        return []

    def tick_step(self, state) -> List[TickResult]:
        return []

    def tick_post_step(self, state) -> List[TickResult]:
        return []

    def on_check_done(self, state) -> List[DoneResult]:
        return []


class MaintainerTest(Test):

    def __init__(self):
        """
        Tests whether the maintainer performs the correct actions and whether his actions register correctly in the env.
        """
        super().__init__()
        self.temp_state_dict = {}
        pass

    def tick_step(self, state) -> List[TickResult]:
        for maintainer in state.entities[M.MAINTAINERS]:

            # has valid actionresult
            self.assertIsInstance(maintainer.state, ActionResult)
            self.assertEqual(maintainer.state.validity, True)

            # will open doors when standing in front
            if maintainer._closed_door_in_path(state):
                self.assertEqual(maintainer.get_move_action(state).name, 'use_door')

            elif maintainer._path and len(maintainer._path) > 1:
                # can move
                print(maintainer.move(maintainer._path[1], state))
                # self.assertTrue(maintainer.move(maintainer._path[1], state))

            if maintainer._next and not maintainer._path:
                # finds valid targets when at target location
                route = maintainer.calculate_route(maintainer._last[-1], state.floortile_graph)
                if entities_at_target_location := [entity for entity in state.entities.by_pos(route[-1])]:
                    self.assertTrue(any(isinstance(e, Machine) for e in entities_at_target_location))
        return []

    def tick_post_step(self, state) -> List[TickResult]:
        for maintainer in state.entities[M.MAINTAINERS]:
            if maintainer._path:
                # was door opened successfully?
                if maintainer._closed_door_in_path(state):
                    door = next(
                        (entity for entity in state.entities.by_pos(maintainer._path[0]) if isinstance(entity, Door)),
                        None)
                    # self.assertEqual(door.is_open, True)

                # when stepping off machine, did maintain action work?
        return []

    def on_check_done(self, state) -> List[DoneResult]:
        for maintainer in state.entities[M.MAINTAINERS]:
            temp_state = maintainer._status
            self.temp_state_dict[maintainer.identifier] = temp_state
        print(self.temp_state_dict)
        return []


class DirtAgentTest(Test):

    def __init__(self):
        """
        Tests whether the dirt agent will perform the correct actions and whether the actions register correctly in the
        environment.
        """
        super().__init__()
        pass

    def on_init(self, state, lvl_map):
        return []

    def on_reset(self):
        return []

    def tick_step(self, state) -> List[TickResult]:
        for agent in state.entities[c.AGENT]:
            print(agent)
            # has valid actionresult
            self.assertIsInstance(agent.state, ActionResult)
            self.assertEqual(agent.state.validity, True)

        return []

    def tick_post_step(self, state) -> List[TickResult]:
        # action success?
        # collisions? if yes, reported?
        return []

    def on_check_done(self, state) -> List[DoneResult]:
        return []

# class ItemAgentTest(Test):
