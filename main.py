from src.Simulator import Simulator

sim = Simulator(use_rt1=True)
sim.set_policy('rt1')
sim.action_spec
sim.set_natural_language_command('move the red block to the right')
sim.get_model_input_format()
sim.run_sim(duration=2)