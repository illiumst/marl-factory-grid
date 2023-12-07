import unittest
from typing import List

import marl_factory_grid.modules.maintenance.constants as M
from marl_factory_grid.algorithms.static.TSP_dirt_agent import TSPDirtAgent
from marl_factory_grid.environment.entity.agent import Agent
from marl_factory_grid.modules import Door, Machine, DirtPile, Item, DropOffLocation, ItemAction
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
            # self.assertEqual(maintainer.state.validity, True)
            # print(f"state validity {maintainer.state.validity}")

            # will open doors when standing in front
            if maintainer._closed_door_in_path(state):
                self.assertEqual(maintainer.get_move_action(state).name, 'use_door')

            # if maintainer._next and not maintainer._path:
            # finds valid targets when at target location
            # route = maintainer.calculate_route(maintainer._last[-1], state.floortile_graph)
            # if entities_at_target_location := [entity for entity in state.entities.by_pos(route[-1])]:
            #     self.assertTrue(any(isinstance(e, Machine) for e in entities_at_target_location))
        return []

    def tick_post_step(self, state) -> List[TickResult]:
        # do maintainers' actions have correct effects on environment i.e. doors open, machines heal
        for maintainer in state.entities[M.MAINTAINERS]:
            if maintainer._path and self.temp_state_dict != {}:
                last_action = self.temp_state_dict[maintainer.identifier]
                if last_action.identifier == 'DoorUse':
                    if door := next((entity for entity in state.entities.get_entities_near_pos(maintainer.pos) if
                                     isinstance(entity, Door)), None):
                        self.assertTrue(door.is_open)
                if last_action.identifier == 'MachineAction':
                    if machine := next((entity for entity in state.entities.get_entities_near_pos(maintainer.pos) if
                                        isinstance(entity, Machine)), None):
                        self.assertEqual(machine.health, 100)
        return []

    def on_check_done(self, state) -> List[DoneResult]:
        for maintainer in state.entities[M.MAINTAINERS]:
            temp_state = maintainer._status
            self.temp_state_dict[maintainer.identifier] = temp_state
        return []


class DirtAgentTest(Test):

    def __init__(self):
        """
        Tests whether the dirt agent will perform the correct actions and whether the actions register correctly in the
        environment.
        """
        super().__init__()
        self.temp_state_dict = {}
        pass

    def on_init(self, state, lvl_map):
        return []

    def on_reset(self):
        return []

    def tick_step(self, state) -> List[TickResult]:
        for dirtagent in [a for a in state.entities[c.AGENT] if "Clean" in a.identifier]:  # isinstance TSPDirtAgent
            # has valid actionresult
            self.assertIsInstance(dirtagent.state, ActionResult)
            # self.assertEqual(agent.state.validity, True)
            # print(f"state validity {maintainer.state.validity}")

        return []

    def tick_post_step(self, state) -> List[TickResult]:
        # do agents' actions have correct effects on environment i.e. doors open, dirt is cleaned
        for dirtagent in [a for a in state.entities[c.AGENT] if "Clean" in a.identifier]:  # isinstance TSPDirtAgent
            if self.temp_state_dict != {}:  # and
                last_action = self.temp_state_dict[dirtagent.identifier]
                if last_action.identifier == 'DoorUse':
                    if door := next((entity for entity in state.entities.get_entities_near_pos(dirtagent.pos) if
                                     isinstance(entity, Door)), None):
                        self.assertTrue(door.is_open)  # TODO catch if someone closes a door in same moment
                if last_action.identifier == 'Clean':
                    if dirt := next((entity for entity in state.entities.get_entities_near_pos(dirtagent.pos) if
                                     isinstance(entity, DirtPile)), None):
                        # print(f"dirt left on pos: {dirt.amount}")
                        self.assertTrue(
                            dirt.amount < 5)  # TODO amount one step before - clean amount?
        return []

    def on_check_done(self, state) -> List[DoneResult]:
        for dirtagent in [a for a in state.entities[c.AGENT] if "Clean" in a.identifier]:  # isinstance TSPDirtAgent
            temp_state = dirtagent._status
            self.temp_state_dict[dirtagent.identifier] = temp_state
        return []


class ItemAgentTest(Test):

    def __init__(self):
        """
        Tests whether the dirt agent will perform the correct actions and whether the actions register correctly in the
        environment.
        """
        super().__init__()
        self.temp_state_dict = {}
        pass

    def on_init(self, state, lvl_map):
        return []

    def on_reset(self):
        return []

    def tick_step(self, state) -> List[TickResult]:
        for itemagent in [a for a in state.entities[c.AGENT] if "Item" in a.identifier]:  # isinstance TSPItemAgent
            # has valid actionresult
            self.assertIsInstance(itemagent.state, ActionResult)
            # self.assertEqual(agent.state.validity, True)
            # print(f"state validity {maintainer.state.validity}")

        return []

    def tick_post_step(self, state) -> List[TickResult]:
        # do agents' actions have correct effects on environment i.e. doors open, items are picked up and dropped off
        for itemagent in [a for a in state.entities[c.AGENT] if "Item" in a.identifier]:  # isinstance TSPItemAgent

            if self.temp_state_dict != {}:  # and
                last_action = self.temp_state_dict[itemagent.identifier]
                if last_action.identifier == 'DoorUse':
                    if door := next((entity for entity in state.entities.get_entities_near_pos(itemagent.pos) if
                                     isinstance(entity, Door)), None):
                        self.assertTrue(door.is_open)
                if last_action.identifier == 'ItemAction':

                    print(last_action.valid_drop_off_reward) #  kann man das nehmen für dropoff vs pickup?
                    # valid pickup?

                    # If it was a pick-up action
                    nearby_items = [e for e in state.entities.get_entities_near_pos(itemagent.pos) if
                                    isinstance(e, Item)]
                    self.assertNotIn(Item, nearby_items)

                    # If the agent has the item in its inventory
                    self.assertTrue(itemagent.bound_entity)

                    # If it was a drop-off action
                    nearby_drop_offs = [e for e in state.entities.get_entities_near_pos(itemagent.pos) if
                                        isinstance(e, DropOffLocation)]
                    if nearby_drop_offs:
                        dol = nearby_drop_offs[0]
                        self.assertTrue(dol.bound_entity)  # item in drop-off location?

                        # Ensure the item is not in the inventory after dropping off
                        self.assertNotIn(Item, state.entities.get_entities_near_pos(itemagent.pos))

        return []

    def on_check_done(self, state) -> List[DoneResult]:
        for itemagent in [a for a in state.entities[c.AGENT] if "Item" in a.identifier]:  # isinstance TSPItemAgent
            temp_state = itemagent._status
            self.temp_state_dict[itemagent.identifier] = temp_state
        return []
