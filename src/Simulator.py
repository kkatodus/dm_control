from dm_control import suite
from dm_control import viewer
import copy
import numpy as np
import tensorflow as tf

import matplotlib
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from IPython.display import HTML
from tf_agents.policies import py_tf_eager_policy


def display_video(frames, framerate=30):
    height, width, _ = frames[0].shape
    dpi = 70
    orig_backend = matplotlib.get_backend()
    matplotlib.use('Agg')  # Switch to headless 'Agg' to inhibit figure rendering.
    fig, ax = plt.subplots(1, 1, figsize=(width / dpi, height / dpi), dpi=dpi)
    matplotlib.use(orig_backend)  # Switch back to the original backend.
    ax.set_axis_off()
    ax.set_aspect('equal')
    ax.set_position([0, 0, 1, 1])
    im = ax.imshow(frames[0])
    def update(frame):
      im.set_data(frame)
      return [im]
    interval = 1000/framerate
    anim = animation.FuncAnimation(fig=fig, func=update, frames=frames,
                                   interval=interval, blit=True, repeat=False)
    return HTML(anim.to_html5_video())


def translate_rt1_action_to_sim_actions(rt1_action):
	base_displacement_vector = rt1_action['action/base_displacement_vector']
	base_rotation = rt1_action['action/base_displacement_vertical_rotation']
	gripper_closedness = rt1_action['action/gripper_closedness_action']
	rotation_delta = rt1_action['action/rotation_delta']
	terminate_episode = rt1_action['action/terminate_episode']
	world_vector = rt1_action['action/world_vector']

	discounted_return = rt1_action['info/discounted_return']
	return_info = rt1_action['info/return']
	action_tokens = rt1_action['state/action_tokens']
	print('action_tokens', action_tokens)
	image = rt1_action['state/image']
	step_num = rt1_action['state/step_num']
	t = rt1_action['state/t']
	return 'something'



class Simulator:
	def __init__(self, domain_name='stretch3', task_name='test', use_rt1=False):
		env = suite.load(domain_name=domain_name, task_name=task_name)
		action_spec = env.action_spec()
		self.env = env
		self.action_spec = action_spec
		self.random_state = np.random.RandomState(42)
		self.id2policy = {
			'random':self.policy_random,
			'rt1':self.policy_rt1
		}
		self.policy = None
		self.model = None
		self.action = None
		if use_rt1:
			model = tf.saved_model.load( './trained_checkpoints/rt1simreal/', tags=None, options=None)
			self.model = model
			self.action = model.signatures['action']

	def get_env_action_spec(self):
		return self.action_spec
		
	def set_policy(self, policy_name):
		self.policy = self.id2policy[policy_name]

	def check_policy(self):
		if not self.policy:
			raise ValueError('Please set the policy using Simulator.set_policy(policy_name)')


	def run_sim(self, duration=4):
		frames = []
		rewards = []
		observations = []
		self.check_policy()
		timestep = self.env.reset()
		
		while self.env.physics.data.time < duration:
			action = self.policy(timestep)
			timestep = self.env.step(action)
			# could access number of cameras by env.physics.model.ncam
			camera = self.env.physics.render(height=200, width=200)
			camera0 = self.env.physics.render(camera_id=0, height=200, width=200)
			camera1 = self.env.physics.render(camera_id=3, height=200, width=200)
			rewards.append(timestep.reward)
			observations.append(copy.deepcopy(timestep.observation))
			frames.append(np.hstack((camera0, camera1, camera)))
		html_video = display_video(frames, framerate=1./self.env.control_timestep())
		return html_video
	
	def launch_viewer(self):
		self.check_policy()
		viewer.launch(self.env, policy=self.policy)
	
#####REGISTER POLICIES HERE#####
	
	def policy_random(self, timestep):
		# we don't use the observation in this policy, we just return a random action at every moment
		del timestep
		action = self.random_state.uniform(self.action_spec.minimum, self.action_spec.maximum, self.action_spec.shape)
		return action
	
	def policy_rt1(self, timestep):
		wrist_image = timestep.observation['wrist_feed']
		head_image = timestep.observation['head_feed']
		batch = 6
		sample_input = {
			"arg_0_discount": tf.constant(0.5, shape=[1]),
			"arg_0_observation_base_pose_tool_reached": tf.constant(0.5, shape=[batch,7]),
			"arg_0_observation_gripper_closed": tf.constant(0.5, shape=[batch, 1]),
			"arg_0_observation_gripper_closedness_commanded": tf.constant(0.5, shape=[batch, 1]),
			"arg_0_observation_height_to_bottom": tf.constant(0.5, shape=[batch, 1]),
			"arg_0_observation_image": tf.constant(1, shape=[batch, 256, 320, 3], dtype=tf.uint8),
			"arg_0_observation_natural_language_embedding": tf.constant(0.5, shape=[batch,512]),
			"arg_0_observation_natural_language_instruction":['something']*batch,
			"arg_0_observation_orientation_box":tf.constant(0.5, shape=[batch,2,3]),
			"arg_0_observation_orientation_start": tf.constant(0.5, shape=[batch,4]),
			"arg_0_observation_robot_orientation_positions_box":tf.constant(0.5, shape=[batch,3,3]), 
			"arg_0_observation_rotation_delta_to_go": tf.constant(0.5, shape=[batch,3]),
			"arg_0_observation_src_rotation": tf.constant(0.5, shape=[batch,4]),
			"arg_0_observation_vector_to_go":tf.constant(0.5, shape=[batch,3]),
			"arg_0_observation_workspace_bounds": tf.constant(0.5, shape=[batch,3, 3]),
			"arg_0_reward": tf.constant(0.5, shape=[1]),
			"arg_0_step_type": tf.constant(1, shape=[1,], dtype=tf.int32),
			"arg_1_action_tokens": tf.constant(1, shape=[batch,batch,11,1,1]),
			"arg_1_image":tf.constant(1, shape=[batch, 6, 256, 320, 3], dtype=tf.uint8),
			"arg_1_step_num": tf.constant(1, shape=[1, 1, 1, 1, 1]),
			"arg_1_t":tf.constant(1, shape=[1, 1, 1, 1, 1], dtype=tf.int32),
		}
		model_action = self.action(**sample_input)
		action = translate_rt1_action_to_sim_actions(model_action)
		action = self.random_state.uniform(self.action_spec.minimum, self.action_spec.maximum, self.action_spec.shape)
		return action




