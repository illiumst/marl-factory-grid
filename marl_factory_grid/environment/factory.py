import shutil

from collections import defaultdict
from itertools import chain
from os import PathLike
from pathlib import Path
from typing import Union, List

import gymnasium as gym
import numpy as np

from marl_factory_grid.utils.level_parser import LevelParser
from marl_factory_grid.utils.observation_builder import OBSBuilder
from marl_factory_grid.utils.config_parser import FactoryConfigParser
from marl_factory_grid.utils import helpers as h
import marl_factory_grid.environment.constants as c
from marl_factory_grid.utils.results import Result

from marl_factory_grid.utils.states import Gamestate


class Factory(gym.Env):

    @property
    def action_space(self):
        """
        TODO


        :return:
        """
        return self.state[c.AGENT].action_space

    @property
    def named_action_space(self):
        """
        TODO


        :return:
        """
        return self.state[c.AGENT].named_action_space

    @property
    def observation_space(self):
        """
        TODO


        :return:
        """
        return self.obs_builder.observation_space(self.state)

    @property
    def named_observation_space(self):
        """
        TODO


        :return:
        """
        return self.obs_builder.named_observation_space(self.state)

    @property
    def params(self) -> dict:
        """
        FIXME LAGEGY


        :return:
        """
        import yaml
        config_path = Path(self._config_file)
        config_dict = yaml.safe_load(config_path.open())
        return config_dict

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __init__(self, config_file: Union[str, PathLike], custom_modules_path: Union[None, PathLike] = None,
                 custom_level_path: Union[None, PathLike] = None):
        """
        TODO


        :return:
        """
        self._config_file = config_file
        self.conf = FactoryConfigParser(self._config_file, custom_modules_path)
        # Attribute Assignment
        if custom_level_path is not None:
            self.level_filepath = Path(custom_level_path)
        else:
            self.level_filepath = Path(__file__).parent.parent / h.LEVELS_DIR / f'{self.conf.level_name}.txt'

        parsed_entities = self.conf.load_entities()
        self.map = LevelParser(self.level_filepath, parsed_entities, self.conf.pomdp_r)

        # Init for later usage:
        # noinspection PyTypeChecker
        self.state: Gamestate = None
        # noinspection PyTypeChecker
        self.obs_builder: OBSBuilder = None

        # expensive - don't use; unless required !
        self._renderer = None

        # Init entities
        entities = self.map.do_init()

        # Init rules
        env_rules = self.conf.load_env_rules()
        entity_rules = self.conf.load_entity_spawn_rules(entities)
        env_rules.extend(entity_rules)

        # Parse the agent conf
        parsed_agents_conf = self.conf.parse_agents_conf()
        self.state = Gamestate(entities, parsed_agents_conf, env_rules, self.map.level_shape,
                               self.conf.env_seed, self.conf.verbose)

        # All is set up, trigger additional init (after agent entity spawn etc)
        self.state.rules.do_all_init(self.state, self.map)

        self.obs_builder = OBSBuilder(self.map.level_shape, self.state, self.map.pomdp_r)

    def __getitem__(self, item):
        return self.state.entities[item]

    def reset(self) -> (dict, dict):

        # Reset information the state holds
        self.state.reset()

        # Reset Information the GlobalEntity collection holds.
        self.state.entities.reset()

        # All is set up, trigger entity spawn with variable pos
        self.state.rules.do_all_reset(self.state)

        # Build initial observations for all agents
        self.obs_builder.reset(self.state)
        return self.obs_builder.build_for_all(self.state)

    def manual_step_init(self) -> List[Result]:
        self.state.curr_step += 1

        # Main Agent Step
        pre_step_result = self.state.rules.tick_pre_step_all(self)
        self.obs_builder.reset(self.state)
        return pre_step_result

    def manual_get_named_agent_obs(self, agent_name: str) -> (List[str], np.ndarray):
        agent = self[c.AGENT][agent_name]
        assert agent, f'"{agent_name}" could not be found. Check the spelling!'
        return self.obs_builder.build_for_agent(agent, self.state)

    def manual_get_agent_obs(self, agent_name: str) -> np.ndarray:
        return self.manual_get_named_agent_obs(agent_name)[1]

    def manual_agent_tick(self, agent_name: str, action: int) -> Result:
        agent = self[c.AGENT][agent_name].clear_temp_state()
        action = agent.actions[action]
        action_result = action.do(agent, self)
        agent.set_state(action_result)
        return action_result

    def manual_finalize_init(self):
        results = list()
        results.extend(self.state.rules.tick_step_all(self))
        results.extend(self.state.rules.tick_post_step_all(self))
        return results

    # Finalize
    def manual_step_finalize(self, tick_result) -> (float, bool, dict):
        # Check Done Conditions
        done_results = self.state.check_done()
        reward, reward_info, done = self.summarize_step_results(tick_result, done_results)

        info = reward_info
        info.update(step_reward=sum(reward), step=self.state.curr_step)
        return reward, done, info

    def step(self, actions):

        if not isinstance(actions, list):
            actions = [int(actions)]

        # --> Action

        # Apply rules, do actions, tick the state, etc...
        tick_result = self.state.tick(actions)

        # Check Done Conditions
        done_results = self.state.check_done()

        # Finalize
        reward, reward_info, done = self.summarize_step_results(tick_result, done_results)

        info = dict(reward_info)

        info.update(step_reward=sum(reward), step=self.state.curr_step)

        obs = self.obs_builder.build_for_all(self.state)
        return None, [x for x in obs.values()], reward, done, info

    def summarize_step_results(self, tick_results: list, done_check_results: list) -> (int, dict, bool):
        # Returns: Reward, Info
        rewards = defaultdict(lambda: 0.0)

        # Gather per agent environment rewards and
        # Combine Info dicts into a global one
        combined_info_dict = defaultdict(lambda: 0.0)
        for result in chain(tick_results, done_check_results):
            if not result:
                raise ValueError()
            if result.reward is not None:
                try:
                    rewards[result.entity.name] += result.reward
                except AttributeError:
                    rewards['global'] += result.reward
            infos = result.get_infos()
            for info in infos:
                assert isinstance(info.value, (float, int))
                combined_info_dict[info.identifier] += info.value

        # Check Done Rule Results
        try:
            done_reason = next(x for x in done_check_results if x.validity)
            done = True
            self.state.print(f'Env done, Reason: {done_reason.identifier}.')
        except StopIteration:
            done = False

        if self.conf.individual_rewards:
            global_rewards = rewards['global']
            del rewards['global']
            reward = [rewards[agent.name] for agent in self.state[c.AGENT]]
            reward = [x + global_rewards for x in reward]
            self.state.print(f"Individual rewards are {dict(rewards)}")
            return reward, combined_info_dict, done
        else:
            reward = sum(rewards.values())
            self.state.print(f"reward is {reward}")
        return reward, combined_info_dict, done

    # noinspection PyGlobalUndefined
    def render(self, mode='human'):
        if not self._renderer:  # lazy init
            from marl_factory_grid.utils.renderer import Renderer
            global Renderer
            self._renderer = Renderer(self.map.level_shape,  view_radius=self.conf.pomdp_r, fps=10)

        render_entities = self.state.entities.render()
        if self.conf.pomdp_r:
            for render_entity in render_entities:
                if render_entity.name == c.AGENT:
                    render_entity.aux = self.obs_builder.curr_lightmaps[render_entity.real_name]
        return self._renderer.render(render_entities)

    def summarize_header(self):
        header = {'rec_step': self.state.curr_step}
        for entity_group in (x for x in self.state if x.name in ['Walls', 'DropOffLocations', 'ChargePods']):
            header.update({f'rec{entity_group.name}': entity_group.summarize_states()})
        return header

    def summarize_state(self):
        summary = {'step': self.state.curr_step}

        # Todo: Protobuff Compatibility Section                                      #######
        #  for entity_group in (x for x in self.state if x.name not in [c.WALLS, c.FLOORS]):
        for entity_group in self.state:
            summary.update({entity_group.name.lower(): entity_group.summarize_states()})
        # TODO Section End                                                          ########
        for key in list(summary.keys()):
            if key not in ['step', 'walls', 'doors', 'agents', 'items', 'dirtPiles', 'batteries']:
                del summary[key]
        return summary

    def save_params(self, filepath: Path):
        # noinspection PyProtectedMember
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(self._config_file, filepath)
