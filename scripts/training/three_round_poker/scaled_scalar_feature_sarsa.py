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
from pypokerai.value_function import LinearModelScaledScalarFeaturesValueFunction
from pypokerai.callback import ResetOpponentValueFunction, InitialStateValueRecorder,\
        EpisodeSampler, WeightsAnalyzer


class ApproxActionValueFunction(SarsaApproxActionValueFunction):

    def __init__(self, handicappers=None):
        super(SarsaApproxActionValueFunction, self).__init__()
        self._handicappers = handicappers

    def setup(self):
        self.delegate = LinearModelScaledScalarFeaturesValueFunction(blind_structure, self._handicappers)
        self.delegate.setup()

    def construct_features(self, state, action):
        return self.delegate.construct_features(state, action)

    def approx_predict_value(self, features):
        return self.delegate.approx_predict_value(features)

    def approx_backup(self, features, backup_target, alpha):
        self.delegate.approx_backup(features, backup_target, alpha)

    def visualize_feature_weights(self):
        return self.delegate.visualize_feature_weights()

    def save(self, save_dir_path):
        self.delegate.save(save_dir_path)

    def load(self, load_dir_path):
        self.delegate.load(load_dir_path)

# Setup directory to output learning results
time_stamp = datetime.now().strftime('%m%d_%H_%M_%S')
TRAINING_TITLE = "scaled_scalar_feature_sarsa_trash_%s" % time_stamp
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "results", TRAINING_TITLE)
os.mkdir(OUTPUT_DIR)

# record log on terminal in log file
sys.stdout = Logger(os.path.join(OUTPUT_DIR, "training.log"))

# copy training script to output dir
training_script_output_path = os.path.join(OUTPUT_DIR, os.path.basename(__file__))
shutil.copyfile(__file__, training_script_output_path)

# copy play game script to output dir
play_script_path = os.path.join(root, "scripts", "play_game.py")
play_script_output_path = os.path.join(OUTPUT_DIR, os.path.basename(play_script_path))
shutil.copyfile(play_script_path, play_script_output_path)

# copy initial value plot script to output dir
initial_value_plot_script_path = os.path.join(root, "scripts", "plot_initial_value.py")
initial_value_plot_script_output_path = os.path.join(OUTPUT_DIR, os.path.basename(initial_value_plot_script_path))
shutil.copyfile(initial_value_plot_script_path, initial_value_plot_script_output_path)

# copy episode generator script to output dir
episode_generate_script_path = os.path.join(root, "scripts", "episode_generator.py")
episode_generate_script_output_path = os.path.join(OUTPUT_DIR, os.path.basename(episode_generate_script_path))
shutil.copyfile(episode_generate_script_path, episode_generate_script_output_path)

# copy round-robin match script to output dir
round_robin_script_path = os.path.join(root, "scripts", "round_robin_match.py")
round_robin_script_output_path = os.path.join(OUTPUT_DIR, os.path.basename(round_robin_script_path))
shutil.copyfile(round_robin_script_path, round_robin_script_output_path)

TEST_LENGTH = 1000000

# Setup algorithm
value_func = ApproxActionValueFunction()
task = TexasHoldemTask(final_round=3, scale_reward=True, lose_penalty=True, shuffle_position=True)
task.set_opponent_value_functions([value_func]*9)
policy = EpsilonGreedyPolicy(eps=0.99)
policy.set_eps_annealing(0.99, 0.1, int(TEST_LENGTH*0.8))
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

episode_log_path = os.path.join(OUTPUT_DIR, "episode_log.txt")
episode_sample_interval = 50000
episode_sampler = EpisodeSampler(episode_sample_interval, episode_log_path, my_uuid, show_weights=True)
callbacks.append(episode_sampler)

weights_output_path = os.path.join(OUTPUT_DIR, "weights_analysis.txt")
weights_sample_interval = 50000
weights_analyzer = WeightsAnalyzer(weights_sample_interval, weights_output_path)
callbacks.append(weights_analyzer)

algorithm.run_gpi(TEST_LENGTH, callbacks)

