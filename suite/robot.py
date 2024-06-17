"""Robot Domain."""

import collections

from dm_control import mujoco
from dm_control.rl import control
from dm_control.suite import base
from dm_control.suite import common
from dm_control.utils import containers

_DEFAULT_TIME_LIMIT = 30
_CONTROL_TIMESTEP = .04

SUITE = containers.TaggedTasks()

def get_model_and_assets():
  """Returns a tuple containing the model XML string and a dict of assets."""
  return common.read_model('robot.xml'), common.ASSETS

@SUITE.add('playing')
def test(time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
  """Returns the Test task."""
  physics = Physics.from_xml_string(*get_model_and_assets())
  task = Robot()
  environment_kwargs = environment_kwargs or {}
  return control.Environment(
      physics, task, time_limit=time_limit, control_timestep=_CONTROL_TIMESTEP,
      **environment_kwargs)

class Physics(mujoco.Physics):
  """Physics class"""

class Robot(base.Task):
  def __init__(self, random=None):
    """Initializes an instance of `Robot`."""
    super(Robot, self).__init__(random=random)

  def initialize_episode(self, physics):
    """Sets the state of the environment at the start of each episode."""
    pass
  
  def get_observation(self, physics):
    """Returns either the pure state or a set of egocentric features."""
    obs = collections.OrderedDict()
    return obs

  def get_reward(self, physics):
    """Returns a reward to the agent."""
    return 0