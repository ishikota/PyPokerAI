#!/usr/local/bin/python

# Resolve path configucation
import os
import sys

root = os.path.join(os.path.dirname(__file__), "..", "..", "..")
src_path = os.path.join(root, "pypokerai")
sys.path.append(root)
sys.path.append(src_path)

# Prepare method to start pdb when crashed
import pdb, traceback, sys
def run_insecure_method(f, args):
    try:
        f(*args)
    except:
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)

# Create Loggar class to record the log output on console
class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

import shutil
from datetime import datetime

from kyoka.algorithm.montecarlo import MonteCarlo, MonteCarloApproxActionValueFunction
from kyoka.algorithm.q_learning import QLearning, QLearningApproxActionValueFunction
from kyoka.algorithm.sarsa import Sarsa, SarsaApproxActionValueFunction
from kyoka.policy import EpsilonGreedyPolicy
from kyoka.callback import LearningRecorder, ManualInterruption

from pypokerai.task import TexasHoldemTask, blind_structure, my_uuid
from pypokerai.value_function import MLPOneLayerActionRecordScaledScalarFeaturesValueFunction
from pypokerai.callback import ResetOpponentValueFunction, InitialStateValueRecorder,\
        EpisodeSampler, WeightsAnalyzer, TrainingLossRecorder, FinishTimeCalculator

FINAL_ROUND = 1
NB_UNIT = 128
class ApproxActionValueFunction(SarsaApproxActionValueFunction):

    def __init__(self, handicappers=None):
        super(SarsaApproxActionValueFunction, self).__init__()
        self._handicappers = handicappers

    def setup(self):
        self.delegate = MLPOneLayerActionRecordScaledScalarFeaturesValueFunction(NB_UNIT, blind_structure, self._handicappers)
        self.delegate.setup()

    def construct_features(self, state, action):
        return self.delegate.construct_features(state, action)

    def approx_predict_value(self, features):
        return self.delegate.approx_predict_value(features)

    def approx_backup(self, features, backup_target, alpha):
        self.delegate.approx_backup(features, backup_target, alpha)

    def save(self, save_dir_path):
        self.delegate.save(save_dir_path)

    def load(self, load_dir_path):
        self.delegate.load(load_dir_path)

# Setup directory to output learning results
time_stamp = datetime.now().strftime('%m%d_%H_%M_%S')
TRAINING_TITLE = "mlp_1layer_step_learning_final_round_%s_%s" % (FINAL_ROUND, time_stamp)
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "results", TRAINING_TITLE)
os.mkdir(OUTPUT_DIR)

# record log on terminal in log file
sys.stdout = Logger(os.path.join(OUTPUT_DIR, "training.log"))

# copy training script to output dir
training_script_output_path = os.path.join(OUTPUT_DIR, os.path.basename(__file__))
shutil.copyfile(__file__, training_script_output_path)

# copy initial value plot script to output dir
initial_value_plot_script_path = os.path.join(root, "scripts", "plot_initial_value.py")
initial_value_plot_script_output_path = os.path.join(OUTPUT_DIR, os.path.basename(initial_value_plot_script_path))
shutil.copyfile(initial_value_plot_script_path, initial_value_plot_script_output_path)

# copy initial value plot script to output dir
loss_plot_script_path = os.path.join(root, "scripts", "plot_loss_history.py")
loss_plot_script_output_path = os.path.join(OUTPUT_DIR, os.path.basename(initial_value_plot_script_path))
shutil.copyfile(loss_plot_script_path, loss_plot_script_output_path)

# copy episode generator script to output dir
episode_generate_script_path = os.path.join(root, "scripts", "episode_generator.py")
episode_generate_script_output_path = os.path.join(OUTPUT_DIR, os.path.basename(episode_generate_script_path))
shutil.copyfile(episode_generate_script_path, episode_generate_script_output_path)

# copy round-robin match script to output dir
round_robin_script_path = os.path.join(root, "scripts", "round_robin_match.py")
round_robin_script_output_path = os.path.join(OUTPUT_DIR, os.path.basename(round_robin_script_path))
shutil.copyfile(round_robin_script_path, round_robin_script_output_path)

exec_round_robin_script_path = os.path.join(root, "scripts", "exec_round_robin.py")
exec_round_robin_script_output_path = os.path.join(OUTPUT_DIR, os.path.basename(exec_round_robin_script_path))
shutil.copyfile(exec_round_robin_script_path, exec_round_robin_script_output_path)

plot_round_robin_script_path = os.path.join(root, "scripts", "plot_round_robin.py")
plot_round_robin_script_output_path = os.path.join(OUTPUT_DIR, os.path.basename(plot_round_robin_script_path))
shutil.copyfile(plot_round_robin_script_path, plot_round_robin_script_output_path)

TEST_LENGTH = 1000000

# Setup algorithm
value_func = ApproxActionValueFunction()
task = TexasHoldemTask(final_round=FINAL_ROUND, scale_reward=True, lose_penalty=False, shuffle_position=True, action_record=True)
task.set_opponent_value_functions([value_func]*9)
policy = EpsilonGreedyPolicy(eps=1.00)
policy.set_eps_annealing(1.00, 0.1, int(TEST_LENGTH*0.8))
algorithm = Sarsa(gamma=0.99)
algorithm.setup(task, policy, value_func)

# load last training result
LOAD_DIR_NAME = ""
LOAD_DIR_PATH = os.path.join(os.path.dirname(__file__), "results", LOAD_DIR_NAME, "checkpoint", "gpi_finished")
if len(LOAD_DIR_NAME) != 0:
    algorithm.load(LOAD_DIR_PATH)

# Setup callbacks
callbacks = []

save_interval = 50000
save_dir_path = os.path.join(OUTPUT_DIR, "checkpoint")
os.mkdir(save_dir_path)
learning_recorder = LearningRecorder(algorithm, save_dir_path, save_interval)
callbacks.append(learning_recorder)

monitor_file_path = os.path.join(OUTPUT_DIR, "stop.txt")
manual_interruption = ManualInterruption(monitor_file_path)
callbacks.append(manual_interruption)

reset_interval = 50000
def value_func_generator():
    f = ApproxActionValueFunction(value_func.delegate.handicappers)
    f.setup()
    return f
reset_opponent_value_func = ResetOpponentValueFunction(save_dir_path, reset_interval, value_func_generator, reset_policy="random")
callbacks.append(reset_opponent_value_func)

score_output_path = os.path.join(OUTPUT_DIR, "initial_value_transition.csv")
initial_value_scorer = InitialStateValueRecorder(score_output_path)
callbacks.append(initial_value_scorer)

loss_record_path = os.path.join(OUTPUT_DIR, "loss_history.csv")
loss_recorder = TrainingLossRecorder(loss_record_path)
callbacks.append(loss_recorder)

calculation_interval = 50000
finish_time_log_path = os.path.join(OUTPUT_DIR, "finish_time.txt")
finish_time_calculator = FinishTimeCalculator(TEST_LENGTH, calculation_interval, finish_time_log_path)
callbacks.append(finish_time_calculator)

episode_log_path = os.path.join(OUTPUT_DIR, "episode_log.txt")
episode_sample_interval = 50000
episode_sampler = EpisodeSampler(episode_sample_interval, episode_log_path, my_uuid, show_weights=False)
callbacks.append(episode_sampler)

# Setup opponents value functions for Step Learning
import glob
step_learning_checkpoints_path = ""
if len(step_learning_checkpoints_path) != 0:
    files = glob.glob(os.path.join(step_learning_checkpoints_path, "after_*_iteration"))
    assert len(files) != 0
    for f in files:
        output_path = os.path.join(save_dir_path, os.path.basename(f))
        shutil.copytree(f, output_path)
    reset_opponent_value_func.after_update(reset_interval, task, "dummy")

algorithm.run_gpi(TEST_LENGTH, callbacks)

