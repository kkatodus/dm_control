from dm_control import suite
from dm_control import viewer
import copy
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import matplotlib
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from IPython.display import HTML
from tf_agents.policies import py_tf_eager_policy

ACTUATORS = ['forward', 'turn', 'lift', 'arm_extend', 'wrist_yaw', 'wrist_pitch', 'wrist_roll', 'grip', 'head_pan', 'head_tilt']

ACTION_TOKENS = ['mode', 'gripper_x', 'gripper_y', 'gripper_z', 'gripper_roll', 'gripper_pitch', 'gripper_yaw', 'gripper_opening', 'base_x', 'base_y', 'base_yaw']

MAX_TOKEN_VALUE = 256

NEUTRAL_POSITION = int(MAX_TOKEN_VALUE/2)

ACTUATOR_LIMITS = {
	'forward':[-1,1]
}

BATCH = 1

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

def get_rt1_action_val_for_key(rt1_action, action_name):
	if action_name not in ACTION_TOKENS:
		raise ValueError('Invalid action name')
	index = ACTION_TOKENS.index(action_name)
	rt1_action_flattened = tf.reshape(rt1_action[0], [6, 11])
	return rt1_action_flattened[0][index]

def map_token_value_to_mujoco_actuation(token_value, actuator_name):
	limits = ACTUATOR_LIMITS[actuator_name]
	return limits[0] + (limits[1]-limits[0])*token_value/MAX_TOKEN_VALUE



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
		self.universal_sentence_encoder = None
		self.natural_language_command = None
		self.natural_language_command_embedding = None
		self.camera_seq = []
		self.action_token_seq = None
		self.time = 0
		if use_rt1:
			model = tf.saved_model.load( './trained_checkpoints/rt1main/', tags=None, options=None)
			self.model = model
			self.action = model.signatures['action']
			self.universal_sentence_encoder = hub.load('https://tfhub.dev/google/universal-sentence-encoder-large/5')
		
	def get_env_action_spec(self):
		return self.action_spec
		
	def set_policy(self, policy_name):
		self.policy = self.id2policy[policy_name]

	def set_natural_language_command(self, command):
		self.natural_language_command = command
		self.natural_language_command_embedding = self.universal_sentence_encoder([command])

	def check_policy(self):
		if not self.policy:
			raise ValueError('Please set the policy using Simulator.set_policy(policy_name)')
		
	def translate_rt1_action_to_sim_actions(self, rt1_action):
		# base_displacement_vector = rt1_action['action/base_displacement_vector']
		# base_rotation = rt1_action['action/base_displacement_vertical_rotation']
		# gripper_closedness = rt1_action['action/gripper_closedness_action']
		# rotation_delta = rt1_action['action/rotation_delta']
		# terminate_episode = rt1_action['action/terminate_episode']
		# world_vector = rt1_action['action/world_vector']

		# discounted_return = rt1_action['info/discounted_return']
		# return_info = rt1_action['info/return']
		action_tokens = rt1_action['state/action_tokens']
		base_forward = get_rt1_action_val_for_key(action_tokens, 'base_y')
		forward_actuator_idx = ACTUATORS.index('forward')
		
		# image = rt1_action['state/image']
		# step_num = rt1_action['state/step_num']
		# t = rt1_action['state/t']
		action = [0]*10
		action[forward_actuator_idx] = map_token_value_to_mujoco_actuation(base_forward, 'forward')
		return action

	def run_sim(self, duration=4):
		frames = []
		rewards = []
		observations = []
		self.check_policy()
		timestep = self.env.reset()
		head_camera = timestep.observation['head_feed']
		wrist_camera = timestep.observation['wrist_feed']
		third_person = timestep.observation['third_person']
		#initializing the last 6 images to be the same
		self.camera_seq = [head_camera]*6
		#initializing the last 6 action tokens to be the same
		self.action_token_seq = tf.constant(NEUTRAL_POSITION, shape=[1, 6, 11, 1, 1])

		frames.append(np.hstack((head_camera, wrist_camera, third_person)))
		
		while self.env.physics.data.time < duration:
			action = self.policy(timestep)
			timestep = self.env.step(action)
			# could access number of cameras by env.physics.model.ncam
			head_camera = timestep.observation['head_feed']
			wrist_camera = timestep.observation['wrist_feed']
			third_person = timestep.observation['third_person']
			self.camera_seq.append(head_camera)
			rewards.append(timestep.reward)
			observations.append(copy.deepcopy(timestep.observation))
			frames.append(np.hstack((head_camera, wrist_camera, third_person)))
			self.time += 1

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
		# print('natural_language_command_embedding:', self.natural_language_command_embedding.shape)	
		# print('wrist_image:', wrist_image.shape)
		# print('head_image:', head_image.shape)

		#we need to feed in the last 6 images into rt1
		last_6_images = self.camera_seq[-6:]
		last_6_images_np = np.array(last_6_images)
		last_6_images_tf = tf.convert_to_tensor(last_6_images_np, dtype=tf.uint8)
		last_6_images_tf = tf.expand_dims(last_6_images_tf, axis=0)

		sample_input = {
			"arg_0_discount": tf.constant(0.5, shape=[1]),
			"arg_0_observation_base_pose_tool_reached": tf.constant(0.5, shape=[BATCH,7]),
			"arg_0_observation_gripper_closed": tf.constant(0.5, shape=[BATCH, 1]),
			"arg_0_observation_gripper_closedness_commanded": tf.constant(0.5, shape=[BATCH, 1]),
			"arg_0_observation_height_to_bottom": tf.constant(0.5, shape=[BATCH, 1]),
			"arg_0_observation_image": tf.expand_dims(head_image, axis=0),
			"arg_0_observation_natural_language_embedding": self.natural_language_command_embedding,
			"arg_0_observation_natural_language_instruction":[self.natural_language_command],
			"arg_0_observation_orientation_box":tf.constant(0.5, shape=[BATCH,2,3]),
			"arg_0_observation_orientation_start": tf.constant(0.5, shape=[BATCH,4]),
			"arg_0_observation_robot_orientation_positions_box":tf.constant(0.5, shape=[BATCH,3,3]), 
			"arg_0_observation_rotation_delta_to_go": tf.constant(0.5, shape=[BATCH,3]),
			"arg_0_observation_src_rotation": tf.constant(0.5, shape=[BATCH,4]),
			"arg_0_observation_vector_to_go":tf.constant(0.5, shape=[BATCH,3]),
			"arg_0_observation_workspace_bounds": tf.constant(0.5, shape=[BATCH,3, 3]),
			"arg_0_reward": tf.constant(0.5, shape=[1]),
			"arg_0_step_type": tf.constant(1, shape=[1,], dtype=tf.int32),
			"arg_1_action_tokens": self.action_token_seq,
			"arg_1_image": last_6_images_tf,
			"arg_1_step_num": tf.constant(1, shape=[1, 1, 1, 1, 1]),
			"arg_1_t":tf.constant(self.time, shape=[1, 1, 1, 1, 1], dtype=tf.int32),
		}
		model_action = self.action(**sample_input)
		self.action_token_seq = model_action['state/action_tokens']
		action = self.translate_rt1_action_to_sim_actions(model_action)
		# action = self.random_state.uniform(self.action_spec.minimum, self.action_spec.maximum, self.action_spec.shape)
		return action




