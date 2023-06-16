import shutil

from collections import defaultdict
from itertools import chain
from os import PathLike
from pathlib import Path
from typing import Union

import gymnasium as gym

from environment.utils.level_parser import LevelParser
from environment.utils.observation_builder import OBSBuilder
from environment.utils.config_parser import FactoryConfigParser
from environment.utils import helpers as h
import environment.constants as c

from environment.utils.states import Gamestate

REC_TAC = 'rec_'


class BaseFactory(gym.Env):

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

    @property
    def summarize_header(self):
        summary_dict = self._summarize_state(stateless_entities=True)
        return summary_dict

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __init__(self, config_file: Union[str, PathLike]):
        self._config_file = config_file
        self.conf = FactoryConfigParser(self._config_file)
        # Attribute Assignment
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
        self.state = None

        # Init entity:
        entities = self.map.do_init()

        # Grab all rules:
        rules = self.conf.load_rules()

        # Agents
        # noinspection PyAttributeOutsideInit
        self.state = Gamestate(entities, rules, self.conf.env_seed)

        agents = self.conf.load_agents(self.map.size, self[c.FLOOR].empty_tiles)
        self.state.entities.add_item({c.AGENT: agents})

        # All is set up, trigger additional init (after agent entity spawn etc)
        self.state.rules.do_all_init(self.state)

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
        # TODO:
        #  if self._record_episodes:
        #      info.update(self._summarize_state())

        obs, reset_info = self.obs_builder.refresh_and_build_for_all(self.state)
        info.update(reset_info)
        return None, [x for x in obs.values()], reward, done, info

    def summarize_step_results(self, tick_results: list, done_check_results: list) -> (int, dict, bool):
        # Returns: Reward, Info
        rewards = defaultdict(lambda: 0.0)

        # Gather per agent env rewards and
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
            self.state.print(f'Env done, Reason: {done_reason.name}.')
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

    def start_recording(self):
        self.conf.do_record = True
        return self.conf.do_record

    def stop_recording(self):
        self.conf.do_record = False
        return not self.conf.do_record

    # noinspection PyGlobalUndefined
    def render(self, mode='human'):
        if not self._renderer:  # lazy init
            from environment.utils.renderer import Renderer
            global Renderer
            self._renderer = Renderer(self.map.level_shape,  view_radius=self.conf.pomdp_r, fps=10)

        render_entities = self.state.entities.render()
        if self.conf.pomdp_r:
            for render_entity in render_entities:
                if render_entity.name == c.AGENT:
                    render_entity.aux = self.obs_builder.curr_lightmaps[render_entity.real_name]
        return self._renderer.render(render_entities)

    def _summarize_state(self, stateless_entities=False):
        summary = {f'{REC_TAC}step': self.state.curr_step}

        for entity_group in self.state:
            if entity_group.is_stateless == stateless_entities:
                summary.update({f'{REC_TAC}{entity_group.name}': entity_group.summarize_states()})
        return summary

    def print(self, string):
        if self.conf.verbose:
            print(string)

    def save_params(self, filepath: Path):
        # noinspection PyProtectedMember
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(self._config_file, filepath)
