import os
from dm_control import suite
from dm_control import viewer
import copy
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from IPython.display import HTML
from tf_agents.policies import py_tf_eager_policy
import tf_agents
from tf_agents.trajectories import time_step as ts
from collections import defaultdict
from datetime import datetime
from PIL import Image

ACTUATORS = ['left_wheel_vel', 'right_wheel_vel', 'lift', 'arm', 'wrist_yaw', 'wrist_pitch', 'wrist_roll', 'gripper', 'head_pan', 'head_tilt']
ACTUATOR_LIMITS = {
    'left_wheel_vel':[-6,6],
    'right_wheel_vel':[-6,6],
    # 'lift':[0,1.1],
    'lift':[0,0.8],
    'arm':[0,0.52],
    'wrist_yaw':[-1.39, 4.42],
    'wrist_pitch':[-1.57, 0.56],
    'wrist_roll':[-3.14, 3.14],
    'gripper':[-0.6, 0.6],
    'head_pan':[-4.04, 1.73],
    'head_tilt':[-1.53, 0.79]
}


ACTION_TOKENS = ['mode', 'gripper_x', 'gripper_y', 'gripper_z', 'gripper_roll', 'gripper_pitch', 'gripper_yaw', 'gripper_opening', 'base_x', 'base_y', 'base_yaw']
RT1_RAW_ACTION_KEYS = [
    'base_displacement_vector',
    'gripper_closedness_action',
    'world_vector',
    'base_displacement_vertical_rotation',
    'rotation_delta',
    'terminate_episode'
]

RT1_RAW_ACTION_KEYS2DIMS = {
    'base_displacement_vector':2,
    'gripper_closedness_action':1,
    'world_vector':3,
    'base_displacement_vertical_rotation':1,
    'rotation_delta':3,
    'terminate_episode':3

}


WHEEL_SEPARATION = 0.3153
WHEEL_DIAMETER = 0.1016

MAX_TOKEN_VALUE = 256

NEUTRAL_POSITION = int(MAX_TOKEN_VALUE/2)

BATCH = 1

def display_video(frames, video_path, framerate=30):
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
    anim.save(video_path, fps=framerate)
    return HTML(anim.to_html5_video())

def get_rt1_action_val_for_key(rt1_action, action_name):
    if action_name not in ACTION_TOKENS:
        raise ValueError('Invalid action name')
    index = ACTION_TOKENS.index(action_name)
    rt1_action_flattened = tf.reshape(rt1_action[0], [6, 11])
    return rt1_action_flattened[0][index]


def map_RT1_output_to_mujoco_limits(rt1_action, mujoco_limits, rt1_limits=[-1, 1]):
    return mujoco_limits[0] + (mujoco_limits[1]-mujoco_limits[0])*(rt1_action-rt1_limits[0])/(rt1_limits[1]-rt1_limits[0])


class Simulator:
    def __init__(self, domain_name='stretch3', task_name='test', use_rt1=False, use_phi3=False, output_dir=None, use_openvla=False):
        env = suite.load(domain_name=domain_name, task_name=task_name)
        action_spec = env.action_spec()
        self.env = env
        self.action_spec = action_spec
        # self.random_state = np.random.RandomState(42)
        self.output_dir = output_dir
        self.output_dir_for_run = os.path.join(output_dir, datetime.now().strftime("%Y%m%d-%H%M%S"))
        os.makedirs(self.output_dir_for_run, exist_ok=True) 
        self.policy = None
        self.model = None
        self.action = None
        self.universal_sentence_encoder = None
        self.natural_language_command = None
        self.natural_language_command_embedding = None
        self.camera_seq = []
        self.action_token_seq = None
        self.time = 0
        self.id2policy = {
            'random': self.policy_random,
            'rt1': self.policy_rt1
        }
        self.rt1_action_records = defaultdict(list)
        self.mujoco_action_records = defaultdict(list)
        self.input_camera_type = None
        self.input_camera_options = ['head_feed', 'wrist_feed', 'third_person', 'forth_person']
        self.half_camera_options = ['head_feed_half', 'wrist_feed_half', 'forth_person_half']
        self.rt1_observation = None
        self.rt1_policy_state = None
        self.multi_camera = None
        self.rt1_camera_seq = []

        if use_rt1:
            model_path = './trained_checkpoints/rt1main/'
            model_path = './rt_1_x_tf_trained_for_002272480_step'
            tfa_policy = py_tf_eager_policy.SavedModelPyTFEagerPolicy(
                model_path=model_path,
                load_specs_from_pbtxt=True,
                use_tf_function=True)
            self.tfa_policy = tfa_policy
            self.rt1_policy_state = self.tfa_policy.get_initial_state(batch_size=1)
            self.rt1_observation = tf_agents.specs.zero_spec_nest(tf_agents.specs.from_spec(self.tfa_policy.time_step_spec.observation))

            self.universal_sentence_encoder = hub.load('https://tfhub.dev/google/universal-sentence-encoder-large/5')

###############UTILS#################################

    def diff_drive_inv_kinematics(self,V:float,omega:float)->tuple:
        #COPIED FROM STRETCH MUJOCO REPO
        """
        Calculate the rotational velocities of the left and right wheels for a differential drive robot."""
        R =WHEEL_DIAMETER/ 2
        L = WHEEL_SEPARATION
        if R <= 0:
            raise ValueError("Radius must be greater than zero.")
        if L <= 0:
            raise ValueError("Distance between wheels must be greater than zero.")
        
        # Calculate the rotational velocities of the wheels
        w_left = (V - (omega * L / 2)) / R
        w_right = (V + (omega * L / 2)) / R
    
        return (w_left, w_right)
        
    def get_env_action_spec(self):
        return self.action_spec

    def set_policy(self, policy_name):
        self.policy = self.id2policy[policy_name]

    def set_camera_type(self, camera_type):
        if camera_type not in self.input_camera_options:
            raise ValueError(f'Invalid camera type. Please choose from: {" ".join(self.input_camera_options)}')
        if self.multi_camera:
            raise ValueError('Please use either set_camera_type or set_multi_camera, not both')
        self.input_camera_type = camera_type

    def set_multi_camera(self, camera_types):
        if not all([camera_type in self.half_camera_options for camera_type in camera_types]):
            raise ValueError(f'Invalid camera type. Please choose from: {" ".join(self.half_camera_options)}')
        if self.input_camera_type:
            raise ValueError('Please use either set_camera_type or set_multi_camera, not both')
        self.multi_camera = camera_types

    def set_natural_language_command(self, command):
        self.natural_language_command = command
        self.natural_language_command_embedding = self.universal_sentence_encoder([command])

    def check_policy(self):
        if not self.policy:
            raise ValueError('Please set the policy using Simulator.set_policy(policy_name)')
        
#############RT1 to Sim Interface ##############################
        
    def translate_rt1_action_to_sim_actions(self, rt1_action):
        # RT1 outputs decomposed
        base_displacement_vector = rt1_action['base_displacement_vector']
        base_displacement_vector_x = base_displacement_vector[0]
        base_displacement_vector_y = base_displacement_vector[1]
        omega = np.arctan2(base_displacement_vector_y, base_displacement_vector_x)

        base_rotation = rt1_action['base_displacement_vertical_rotation']
        gripper_closedness = rt1_action['gripper_closedness_action']
        rotation_delta = rt1_action['rotation_delta']
        rt1_roll = rotation_delta[0]
        rt1_pitch = rotation_delta[1]
        rt1_yaw = rotation_delta[2]

        terminate_episode = rt1_action['terminate_episode']
        world_vector = rt1_action['world_vector']
        world_vector_x = world_vector[0]
        world_vector_y = world_vector[1]
        world_vector_z = world_vector[2]

        world_vector_r = np.sqrt(world_vector_x**2 + world_vector_y**2)
        world_vector_theta = np.arctan2(world_vector_y, world_vector_x) + np.pi/2
        #Mujoco actuator indices		
        head_tilt_actuator_idx = ACTUATORS.index('head_tilt')
        head_pan_actuator_idx = ACTUATORS.index('head_pan')
        left_wheel_vel_actuator_idx = ACTUATORS.index('left_wheel_vel')
        right_wheel_vel_actuator_idx = ACTUATORS.index('right_wheel_vel')
        lift_actuator_idx = ACTUATORS.index('lift')
        arm_actuator_idx = ACTUATORS.index('arm')
        wrist_yaw_actuator_idx = ACTUATORS.index('wrist_yaw')
        wrist_pitch_actuator_idx = ACTUATORS.index('wrist_pitch')
        wrist_roll_actuator_idx = ACTUATORS.index('wrist_roll')
        gripper_actuator_idx = ACTUATORS.index('gripper')

        #Mappin to mujoco actuations
        w_left, w_right = self.diff_drive_inv_kinematics(0, world_vector_theta)
        lift_height = map_RT1_output_to_mujoco_limits(world_vector_z, ACTUATOR_LIMITS['lift'])
        arm_extend = map_RT1_output_to_mujoco_limits(world_vector_r, ACTUATOR_LIMITS['arm'])
        wrist_pitch = map_RT1_output_to_mujoco_limits(rt1_pitch, ACTUATOR_LIMITS['wrist_pitch'])
        wrist_yaw = map_RT1_output_to_mujoco_limits(rt1_yaw, ACTUATOR_LIMITS['wrist_yaw'])
        wrist_roll = map_RT1_output_to_mujoco_limits(rt1_roll, ACTUATOR_LIMITS['wrist_roll'])
        gripper_opening = map_RT1_output_to_mujoco_limits(gripper_closedness[0], ACTUATOR_LIMITS['gripper'])

        action = [0]*10
        action[head_tilt_actuator_idx] = -1.1
        action[head_pan_actuator_idx] = -1.57
        action[left_wheel_vel_actuator_idx] = w_left
        action[right_wheel_vel_actuator_idx] = w_right
        action[lift_actuator_idx] = lift_height
        action[arm_actuator_idx] = min(arm_extend+0.07, ACTUATOR_LIMITS['arm'][1])
        action[wrist_yaw_actuator_idx] = wrist_yaw
        # action[wrist_yaw_actuator_idx] = 0.05
        # action[wrist_pitch_actuator_idx] = wrist_pitch
        action[wrist_pitch_actuator_idx] = 0.1
        # action[wrist_roll_actuator_idx] = wrist_roll
        action[wrist_roll_actuator_idx] = 0
        action[gripper_actuator_idx] = gripper_opening*-1

        self.record_mujoco_actions(action)

        return action


###############SIMULATION###########################	
    
    def launch_viewer(self):
        self.check_policy()
        viewer.launch(self.env, policy=self.policy)

    def run_sim(self, duration=4):
        frames = []
        timestep = self.env.reset()
        head_camera = timestep.observation['head_feed']
        wrist_camera = timestep.observation['wrist_feed']
        third_person = timestep.observation['third_person']
        forth_person = timestep.observation['forth_person']

        frames.append(np.hstack((head_camera, wrist_camera, third_person, forth_person)))

        looped = -1
        stepgap = 4

        steps = 0
        last_action = None
        while self.env.physics.data.time < duration:
        
            print(self.env.physics.data.time/duration)
            if self.env.physics.data.time/duration == 0.0:
                print('starting')
                looped += 1
            if looped==1:
                break
                
            if steps%stepgap==0:    
                mujoco_action = self.policy(timestep)
            else:
                mujoco_action = last_action
            last_action = mujoco_action
            timestep = self.env.step(mujoco_action)
            #could access number of cameras by env.physics.model.ncam
            head_camera = timestep.observation['head_feed']
            wrist_camera = timestep.observation['wrist_feed']
            third_person = timestep.observation['third_person']
            forth_person = timestep.observation['forth_person']
            
            frames.append(np.hstack((head_camera, wrist_camera, third_person, forth_person)))
            self.rt1_camera_seq.append(self.get_camera_input_from_timestep(timestep))
            steps += 1
        self.plot_rt1_actions()
        self.plot_mujoco_actions()
        html_video = display_video(frames, video_path=os.path.join(self.output_dir_for_run, f'{self.natural_language_command}-sim.mp4'), framerate=1./self.env.control_timestep())
        rt1_video = display_video(self.rt1_camera_seq, video_path=os.path.join(self.output_dir_for_run, f'{self.natural_language_command}-rt1.mp4'), framerate=1./self.env.control_timestep())
        im = Image.fromarray(self.rt1_camera_seq[-1])
        im.save("your_file.jpeg")
        return html_video	
    
    def get_camera_input_from_timestep(self, timestep):
        if self.multi_camera:
            camera_inputs = []
            for camera_type in self.multi_camera:
                camera_inputs.append(timestep.observation[camera_type])
            stacked = np.hstack(camera_inputs)
            return stacked
        else:
            return timestep.observation[self.input_camera_type]

    

#################RECORDING RT1 ACTIONS#####################

    def record_mujoco_actions(self, mujoco_action):
        for idx, action in enumerate(mujoco_action):
            self.mujoco_action_records[ACTUATORS[idx]].append(action)

    def record_rt1_actions(self, rt1_action):
        for action_key in RT1_RAW_ACTION_KEYS:
                one_action = rt1_action[action_key]
                for dim in range(RT1_RAW_ACTION_KEYS2DIMS[action_key]):
                    self.rt1_action_records[action_key+'_'+str(dim)].append(one_action[dim])

    def plot_rt1_actions(self):
        fig, ax = plt.subplots(len(self.rt1_action_records), figsize=(20, 50))
        for idx, (key, value) in enumerate(self.rt1_action_records.items()):
            ax[idx].plot(value)
            ax[idx].set_title(key)
        plt.savefig(os.path.join(self.output_dir_for_run, 'rt1_actions.png'))	

    def plot_mujoco_actions(self):
        fig, ax = plt.subplots(len(self.mujoco_action_records), figsize=(20, 50))
        for idx, (key, value) in enumerate(self.mujoco_action_records.items()):
            ax[idx].plot(value)
            ax[idx].set_title(key)
        plt.savefig(os.path.join(self.output_dir_for_run, 'mujoco_actions.png'))

#####REGISTER POLICIES HERE#####
    
    def policy_random(self, timestep):
        # we don't use the observation in this policy, we just return a random action at every moment
        del timestep
        action = self.random_state.uniform(self.action_spec.minimum, self.action_spec.maximum, self.action_spec.shape)
        return action
    
    def policy_rt1(self, timestep):

        self.rt1_observation['natural_language_embedding'] = self.natural_language_command_embedding
        camera_input = self.get_camera_input_from_timestep(timestep)
        self.rt1_observation['image'] = camera_input
        #experimenting whether feeding both head and wrist camera images to rt1 improves performance
        tfa_time_step = ts.transition(self.rt1_observation, reward=np.zeros((), dtype=np.float32))

        policy_step = self.tfa_policy.action(tfa_time_step, self.rt1_policy_state)
        rt1_action = policy_step.action
        mujoco_action = self.translate_rt1_action_to_sim_actions(rt1_action)
        self.rt1_policy_state = policy_step.state
        self.record_rt1_actions(rt1_action)
        return mujoco_action
    
    # def policy_rt1(self, timestep):
    # 	if not self.natural_language_command:
    # 		raise ValueError('Please set the natural language command using Simulator.set_natural_language_command(command)')
    # 	wrist_image = timestep.observation['wrist_feed']
    # 	head_image = timestep.observation['head_feed']
    # 	third_person = timestep.observation['third_person']
    # 	# print('natural_language_command_embedding:', self.natural_language_command_embedding.shape)	
    # 	# print('wrist_image:', wrist_image.shape)
    # 	# print('head_image:', head_image.shape)

    # 	#we need to feed in the last 6 images into rt1
    # 	last_6_images = self.camera_seq[-6:]
    # 	last_6_images_np = np.array(last_6_images)
    # 	last_6_images_tf = tf.convert_to_tensor(last_6_images_np, dtype=tf.uint8)
    # 	last_6_images_tf = tf.expand_dims(last_6_images_tf, axis=0)

    # 	sample_input = {
    # 		"arg_0_discount": tf.constant(1.0, shape=[1], dtype=tf.float32),
    # 		# "arg_0_observation_base_pose_tool_reached": tf.constant(0.0, shape=[BATCH,7]),
    # 		# "arg_0_observation_gripper_closed": tf.constant(0.5, shape=[BATCH, 1]),
    # 		# "arg_0_observation_gripper_closedness_commanded": tf.constant(0.5, shape=[BATCH, 1]),
    # 		# "arg_0_observation_height_to_bottom": tf.constant(0.5, shape=[BATCH, 1]),
    # 		"arg_0_observation_image": tf.expand_dims(head_image, axis=0),
    # 		"arg_0_observation_natural_language_embedding": self.natural_language_command_embedding,
    # 		"arg_0_observation_natural_language_instruction":[self.natural_language_command],
    # 		# "arg_0_observation_orientation_box":tf.constant(0.5, shape=[BATCH,2,3]),
    # 		# "arg_0_observation_orientation_start": tf.constant(0.5, shape=[BATCH,4]),
    # 		# "arg_0_observation_robot_orientation_positions_box":tf.constant(0.5, shape=[BATCH,3,3]), 
    # 		# "arg_0_observation_rotation_delta_to_go": tf.constant(0.5, shape=[BATCH,3]),
    # 		# "arg_0_observation_src_rotation": tf.constant(0.5, shape=[BATCH,4]),
    # 		# "arg_0_observation_vector_to_go":tf.constant(0.5, shape=[BATCH,3]),
    # 		# "arg_0_observation_workspace_bounds": tf.constant(0.5, shape=[BATCH, 3, 3]),
    # 		"arg_0_reward": tf.constant(0.0, shape=[1], dtype=tf.float32),
    # 		"arg_0_step_type": tf.constant(1, shape=[1,], dtype=tf.int32),
    # 		"arg_1_action_tokens": self.action_token_seq,
    # 		# "arg_1_image": last_6_images_tf,
    # 		"arg_1_seq_idx": tf.constant(0, shape=[BATCH, 1, 1, 1, 1]),
    # 		'arg_1_context_image_tokens': tf.constant(1.0, shape=[BATCH, 15, 81, 1, 512], dtype=tf.float32),
    # 		# "arg_1_step_num": tf.constant(0, shape=[1, 1, 1, 1, 1]),
    # 		# "arg_1_t":tf.constant(self.time, shape=[1, 1, 1, 1, 1], dtype=tf.int32),
    # 	}

    # 	model_action = self.action(**sample_input)
    # 	self.record_rt1_actions(model_action)
    # 	self.action_token_seq = model_action['state/action_tokens']
    # 	action = self.translate_rt1_action_to_sim_actions(model_action)
    # 	# action = self.random_state.uniform(self.action_spec.minimum, self.action_spec.maximum, self.action_spec.shape)
    # 	return action




