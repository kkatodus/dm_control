# Copyright 2020 The dm_control Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================


import collections
import os

from dm_control import mujoco
from dm_control.rl import control
from dm_control.suite import base
from dm_control.suite import common
from dm_control.utils import containers

from lxml import etree

from dm_control.utils import io as resources

_DEFAULT_TIME_LIMIT = 30
_CONTROL_TIMESTEP = .04


SUITE = containers.TaggedTasks()

_ASSET_DIR = os.path.join(os.path.dirname(__file__), 'stretch3/assets')


def make_model():
  xml_string = common.read_model('stretch3_scene.xml')
  parser = etree.XMLParser(remove_blank_text=True)
  mjcf = etree.XML(xml_string, parser)
  return etree.tostring(mjcf, pretty_print=True)

def get_model_and_assets(floor_size=10, remove_ball=True):
  """Returns a tuple containing the model XML string and a dict of assets."""
  assets = common.ASSETS.copy()
  _, _, filenames = next(resources.WalkResources(_ASSET_DIR))
  for filename in filenames:
    assets[filename] = resources.GetResource(os.path.join(_ASSET_DIR, filename))
  return make_model(), assets

@SUITE.add('playing')
def test(time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
  """Returns the Test task."""
  physics = Physics.from_xml_string(*get_model_and_assets())
  
  task = Test()
  environment_kwargs = environment_kwargs or {}
  return control.Environment(
      physics, task, time_limit=time_limit, control_timestep=_CONTROL_TIMESTEP,
      **environment_kwargs)

class Physics(mujoco.Physics):
  """Physics simulation with additional features for the Stretch3 domain.
  We can implement functions to access various observations about the environment which we can pass into the observations dictionary in the Task class implemented below.
  """

class Test(base.Task):
  """"""
  def __init__(self, random=None):
    """Initializes an instance of `Test`."""
    super(Test, self).__init__(random=random)

  def initialize_episode(self, physics):
    """Sets the state of the environment at the start of each episode."""
    pass
  
  def get_observation(self, physics):
    """Returns the state we want to expose the agent to"""
    obs = collections.OrderedDict()
    obs['wrist_feed'] = physics.render(camera_id=0, height=256, width=320)
    obs['head_feed'] = physics.render(camera_id=3, height=256, width=320)
    return obs

  def get_reward(self, physics):
    """Returns a reward to the agent."""
    return 0



