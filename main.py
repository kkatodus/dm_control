import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))
tf.test.gpu_device_name()

from paths import ROOT, OUTPUT_DIR
from src.Simulator import Simulator

# REPETITION = 3
# half_camera_types = ['head_feed_half', 'wrist_feed_half', 'forth_person_half']
# camera_types = ['head_feed', 'wrist_feed', 'third_person', 'forth_person']
# for full_camera_type in camera_types:
# 	for _ in range(REPETITION):
# 		sim = Simulator(use_rt1=True, output_dir=OUTPUT_DIR)
# 		sim.set_natural_language_command('grab the blue box with the gripper')
# 		sim.set_policy('rt1')
# 		sim.set_camera_type(full_camera_type)
# 		sim.run_sim(duration=10)


# half_camera_combos = [
# 	['head_feed_half', 'wrist_feed_half'],
# 	['head_feed_half', 'forth_person_half'],
# 	['wrist_feed_half', 'forth_person_half']
# ]
# for half_camera_combo in half_camera_combos:
# 	for _ in range(REPETITION):
# 		sim = Simulator(use_rt1=True, output_dir=OUTPUT_DIR)
# 		sim.set_natural_language_command('grab the blue box with the gripper')
# 		sim.set_policy('rt1')
# 		sim.multi_camera = half_camera_combo
# 		sim.run_sim(duration=10)


sim = Simulator(use_rt1=True, output_dir=OUTPUT_DIR)
sim.set_natural_language_command('grab the blue box with the gripper and place it on the floor')
sim.set_policy('rt1')
sim.multi_camera = ['head_feed_half', 'wrist_feed_half']
sim.run_sim(duration=60)