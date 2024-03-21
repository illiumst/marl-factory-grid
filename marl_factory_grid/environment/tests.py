import unittest
from typing import List

import marl_factory_grid.modules.maintenance.constants as M
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
        Base test class for unit tests that provides base functions to be overwritten that are automatically called by
        the StepTests class.
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
            self.assertIsInstance(maintainer.state, (ActionResult, TickResult))
            # print(f"state validity maintainer: {maintainer.state.validity}")

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
                if maintainer.identifier in self.temp_state_dict:
                    last_action = self.temp_state_dict[maintainer.identifier]
                    if last_action.identifier == 'DoorUse':
                        if door := next((entity for entity in state.entities.get_entities_near_pos(maintainer.pos) if
                                         isinstance(entity, Door)), None):
                            agents_near_door = [agent for agent in state.entities.get_entities_near_pos(door.pos) if
                                                isinstance(agent, Agent)]
                            if len(agents_near_door) < 2:
                                self.assertTrue(door.is_open)
                    if last_action.identifier == 'MachineAction':
                        if machine := next((entity for entity in state.entities.get_entities_near_pos(maintainer.pos) if
                                            isinstance(entity, Machine)), None):
                            self.assertEqual(machine.health, 100)
        return []

    def on_check_done(self, state) -> List[DoneResult]:
        # clear dict as the maintainer identifier increments each run the dict would fill over episodes
        self.temp_state_dict = {}
        for maintainer in state.entities[M.MAINTAINERS]:
            temp_state = maintainer._status
            if isinstance(temp_state, (ActionResult, TickResult)):
                # print(f"maintainer {temp_state}")
                self.temp_state_dict[maintainer.identifier] = temp_state
            else:
                self.temp_state_dict[maintainer.identifier] = None
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
            # state usually is an actionresult but after a crash, tickresults are reported
            self.assertIsInstance(dirtagent.state, (ActionResult, TickResult))
            # print(f"state validity dirtagent: {dirtagent.state.validity}")
        return []

    def tick_post_step(self, state) -> List[TickResult]:
        # do agents' actions have correct effects on environment i.e. doors open, dirt is cleaned
        for dirtagent in [a for a in state.entities[c.AGENT] if "Clean" in a.identifier]:  # isinstance TSPDirtAgent
            if self.temp_state_dict != {}:
                last_action = self.temp_state_dict[dirtagent.identifier]
                if last_action.identifier == 'DoorUse':
                    if door := next((entity for entity in state.entities.get_entities_near_pos(dirtagent.pos) if
                                     isinstance(entity, Door)), None):
                        agents_near_door = [agent for agent in state.entities.get_entities_near_pos(door.pos) if
                                            isinstance(agent, Agent)]
                        if len(agents_near_door) < 2:
                            # self.assertTrue(door.is_open)
                            if door.is_closed:
                                print("door should be open but seems closed.")
                if last_action.identifier == 'Clean':
                    if dirt := next((entity for entity in state.entities.get_entities_near_pos(dirtagent.pos) if
                                     isinstance(entity, DirtPile)), None):
                        # print(f"dirt left on pos: {dirt.amount}")
                        self.assertTrue(dirt.amount < 5)  # get dirt amount one step before - clean amount
        return []

    def on_check_done(self, state) -> List[DoneResult]:
        for dirtagent in [a for a in state.entities[c.AGENT] if "Clean" in a.identifier]:  # isinstance TSPDirtAgent
            temp_state = dirtagent._status
            if isinstance(temp_state, (ActionResult, TickResult)):
                # print(f"dirtagent {temp_state}")
                self.temp_state_dict[dirtagent.identifier] = temp_state
            else:
                self.temp_state_dict[dirtagent.identifier] = None
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
            # state usually is an actionresult but after a crash, tickresults are reported
            self.assertIsInstance(itemagent.state, (ActionResult, TickResult))
            # self.assertEqual(agent.state.validity, True)
            # print(f"state validity itemagent: {itemagent.state.validity}")

        return []

    def tick_post_step(self, state) -> List[TickResult]:
        # do agents' actions have correct effects on environment i.e. doors open, items are picked up and dropped off
        for itemagent in [a for a in state.entities[c.AGENT] if "Item" in a.identifier]:  # isinstance TSPItemAgent

            if self.temp_state_dict != {}:  # and
                last_action = self.temp_state_dict[itemagent.identifier]
                if last_action.identifier == 'DoorUse':
                    if door := next((entity for entity in state.entities.get_entities_near_pos(itemagent.pos) if
                                     isinstance(entity, Door)), None):
                        agents_near_door = [agent for agent in state.entities.get_entities_near_pos(door.pos) if
                                            isinstance(agent, Agent)]
                        if len(agents_near_door) < 2:
                            # self.assertTrue(door.is_open)
                            if door.is_closed:
                                print("door should be open but seems closed.")

                # if last_action.identifier == 'ItemAction':
                #     If it was a pick-up action the item should be in the agents inventory and not in his neighboring
                #     positions anymore
                #     nearby_items = [e for e in state.entities.get_entities_near_pos(itemagent.pos) if
                #                     isinstance(e, Item)]
                #     self.assertNotIn(Item, nearby_items)
                #     self.assertTrue(itemagent.bound_entity)  # where is the inventory
                #
                #     If it was a drop-off action the item should not be in the agents inventory anymore but instead in
                #     the drop-off locations inventory
                #
                #     if nearby_drop_offs := [e for e in state.entities.get_entities_near_pos(itemagent.pos) if
                #     isinstance(e, DropOffLocation)]:
                #         dol = nearby_drop_offs[0]
                #         self.assertTrue(dol.bound_entity)  # item in drop-off location?
                #         self.assertNotIn(Item, state.entities.get_entities_near_pos(itemagent.pos))

        return []

    def on_check_done(self, state) -> List[DoneResult]:
        for itemagent in [a for a in state.entities[c.AGENT] if "Item" in a.identifier]:  # isinstance TSPItemAgent
            temp_state = itemagent._status
            # print(f"itemagent {temp_state}")
            self.temp_state_dict[itemagent.identifier] = temp_state
        return []


class TargetAgentTest(Test):

    def __init__(self):
        """
        Tests whether the target agent will perform the correct actions and whether the actions register correctly in the
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
        for targetagent in [a for a in state.entities[c.AGENT] if "Target" in a.identifier]:
            # state usually is an actionresult but after a crash, tickresults are reported
            self.assertIsInstance(targetagent.state, (ActionResult, TickResult))
            # print(f"state validity targetagent: {targetagent.state.validity}")
        return []

    def tick_post_step(self, state) -> List[TickResult]:
        # do agents' actions have correct effects on environment i.e. doors open, targets are destinations
        for targetagent in [a for a in state.entities[c.AGENT] if "Target" in a.identifier]:
            if self.temp_state_dict != {}:
                last_action = self.temp_state_dict[targetagent.identifier]
                if last_action.identifier == 'DoorUse':
                    if door := next((entity for entity in state.entities.get_entities_near_pos(targetagent.pos) if
                                     isinstance(entity, Door)), None):
                        agents_near_door = [agent for agent in state.entities.get_entities_near_pos(door.pos) if
                                            isinstance(agent, Agent)]
                        if len(agents_near_door) < 2:
                            # self.assertTrue(door.is_open)
                            if door.is_closed:
                                print("door should be open but seems closed.")

        return []

    def on_check_done(self, state) -> List[DoneResult]:
        for targetagent in [a for a in state.entities[c.AGENT] if "Target" in a.identifier]:
            temp_state = targetagent._status
            # print(f"targetagent {temp_state}")
            self.temp_state_dict[targetagent.identifier] = temp_state
        return []
