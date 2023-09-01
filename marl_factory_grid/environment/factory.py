import shutil

from collections import defaultdict
from itertools import chain
from os import PathLike
from pathlib import Path
from typing import Union

import gymnasium as gym

from marl_factory_grid.utils.level_parser import LevelParser
from marl_factory_grid.utils.observation_builder import OBSBuilder
from marl_factory_grid.utils.config_parser import FactoryConfigParser
from marl_factory_grid.utils import helpers as h
import marl_factory_grid.environment.constants as c

from marl_factory_grid.utils.states import Gamestate


class Factory(gym.Env):

    @property
    def action_space(self):
        return self.state[c.AGENT].action_space

    @property
    def named_action_space(self):
        return self.state[c.AGENT].named_action_space

    @property
    def observation_space(self):
        return self.obs_builder.observation_space(self.state)

    @property
    def named_observation_space(self):
        return self.obs_builder.named_observation_space(self.state)

    @property
    def params(self) -> dict:
        import yaml
        config_path = Path(self._config_file)
        config_dict = yaml.safe_load(config_path.open())
        return config_dict

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __init__(self, config_file: Union[str, PathLike], custom_modules_path: Union[None, PathLike] = None,
                 custom_level_path: Union[None, PathLike] = None):
        self._config_file = config_file
        self.conf = FactoryConfigParser(self._config_file, custom_modules_path)
        # Attribute Assignment
        if custom_level_path is not None:
            self.level_filepath = Path(custom_level_path)
        else:
            self.level_filepath = Path(__file__).parent.parent / h.LEVELS_DIR / f'{self.conf.level_name}.txt'
        self._renderer = None  # expensive - don't use; unless required !

        parsed_entities = self.conf.load_entities()
        self.map = LevelParser(self.level_filepath, parsed_entities, self.conf.pomdp_r)

        # Init for later usage:
        self.state: Gamestate
        self.map: LevelParser
        self.obs_builder: OBSBuilder

        # TODO: Reset ---> document this
        self.reset()

    def __getitem__(self, item):
        return self.state.entities[item]

    def reset(self) -> (dict, dict):
        if hasattr(self, 'state'):
            for entity_group in self.state.entities:
                try:
                    entity_group[0].reset_uid()
                except (AttributeError, TypeError):
                    pass

        self.state = None

        # Init entity:
        entities = self.map.do_init()

        # Grab all )rules:
        rules = self.conf.load_rules()

        # Agents
        # noinspection PyAttributeOutsideInit
        self.state = Gamestate(entities, rules, self.conf.env_seed)

        agents = self.conf.load_agents(self.map.size, self[c.FLOOR].empty_tiles)
        self.state.entities.add_item({c.AGENT: agents})

        # All is set up, trigger additional init (after agent entity spawn etc)
        self.state.rules.do_all_init(self.state, self.map)

        # Observations
        # noinspection PyAttributeOutsideInit
        self.obs_builder = OBSBuilder(self.map.level_shape, self.state, self.map.pomdp_r)
        return self.obs_builder.refresh_and_build_for_all(self.state)

    def step(self, actions):

        if not isinstance(actions, list):
            actions = [int(actions)]

        # Apply rules, do actions, tick the state, etc...
        tick_result = self.state.tick(actions)

        # Check Done Conditions
        done_results = self.state.check_done()

        # Finalize
        reward, reward_info, done = self.summarize_step_results(tick_result, done_results)

        info = reward_info

        info.update(step_reward=sum(reward), step=self.state.curr_step)

        obs, reset_info = self.obs_builder.refresh_and_build_for_all(self.state)
        info.update(reset_info)
        return None, [x for x in obs.values()], reward, done, info

    def summarize_step_results(self, tick_results: list, done_check_results: list) -> (int, dict, bool):
        # Returns: Reward, Info
        rewards = defaultdict(lambda: 0.0)

        # Gather per agent environment rewards and
        # Combine Info dicts into a global one
        combined_info_dict = defaultdict(lambda: 0.0)
        for result in chain(tick_results, done_check_results):
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
            self.state.print(f"rewards are {rewards}")
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
        for entity_group in (x for x in self.state if x.name in ['Walls', 'Floors', 'DropOffLocations', 'ChargePods']):
            header.update({f'rec{entity_group.name}': entity_group.summarize_states()})
        return header

    def summarize_state(self):
        summary = {'step': self.state.curr_step}

        # Todo: Protobuff Compatibility Section                                      #######
        #  for entity_group in (x for x in self.state if x.name not in [c.WALLS, c.FLOORS]):
        for entity_group in (x for x in self.state if x.name not in [c.FLOORS]):
            summary.update({entity_group.name.lower(): entity_group.summarize_states()})
        # TODO Section End                                                          ########
        for key in list(summary.keys()):
            if key not in ['step', 'walls', 'doors', 'agents', 'items', 'dirtPiles', 'batteries']:
                del summary[key]
        return summary

    def print(self, string):
        if self.conf.verbose:
            print(string)

    def save_params(self, filepath: Path):
        # noinspection PyProtectedMember
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(self._config_file, filepath)
