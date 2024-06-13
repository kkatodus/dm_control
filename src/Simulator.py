from dm_control import suite
from dm_control import viewer
import copy
import numpy as np

import matplotlib
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from IPython.display import HTML
import PIL.Image

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


class Simulator:
	def __init__(self, domain_name='stretch3', task_name='test'):
		env = suite.load(domain_name=domain_name, task_name=task_name)
		action_spec = env.action_spec()
		self.env = env
		self.action_spec = action_spec
		self.random_state = np.random.RandomState(42)
		self.id2policy = {
			'random':self.policy_random
		}
		self.policy = None

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
		action = self.random_state.uniform(self.action_spec.minimum, self.action_spec.maximum, self.action_spec.shape)
		return action




