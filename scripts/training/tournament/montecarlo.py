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
from pypokerai.value_function import LinearModelScalarFeaturesValueFunction,\
        LinearModelScaledScalarFeaturesValueFunction, LinearModelOnehotFeaturesValueFunction
from pypokerai.callback import ResetOpponentValueFunction, InitialStateValueRecorder,\
        EpisodeSampler, WeightsAnalyzer


class ApproxActionValueFunction(MonteCarloApproxActionValueFunction):

    def __init__(self, handicappers=None):
        super(MonteCarloApproxActionValueFunction, self).__init__()
        self._handicappers = handicappers

    def setup(self):
        self.delegate = LinearModelOnehotFeaturesValueFunction(blind_structure, self._handicappers)
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
TRAINING_TITLE = "montecarlo_trash_%s" % time_stamp
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "results", TRAINING_TITLE)
os.mkdir(OUTPUT_DIR)

# record log on terminal in log file
sys.stdout = Logger(os.path.join(OUTPUT_DIR, "training.log"))

# copy training script to output dir
script_output_path = os.path.join(OUTPUT_DIR, os.path.basename(__file__))
shutil.copyfile(__file__, script_output_path)

TEST_LENGTH = 10000

# Setup algorithm
value_func = ApproxActionValueFunction()
task = TexasHoldemTask(scale_reward=True, lose_penalty=True)
task.set_opponent_value_functions([value_func]*9)
policy = EpsilonGreedyPolicy(eps=0.99)
policy.set_eps_annealing(0.99, 0.1, TEST_LENGTH)
algorithm = MonteCarlo(gamma=0.99)
algorithm.setup(task, policy, value_func)

# Setup callbacks
callbacks = []

save_interval = 1000
save_dir_path = os.path.join(OUTPUT_DIR, "checkpoint")
os.mkdir(save_dir_path)
learning_recorder = LearningRecorder(algorithm, save_dir_path, save_interval)
callbacks.append(learning_recorder)

monitor_file_path = os.path.join(OUTPUT_DIR, "stop.txt")
manual_interruption = ManualInterruption(monitor_file_path)
callbacks.append(manual_interruption)

reset_interval = 1000
def value_func_generator():
    f = ApproxActionValueFunction(value_func.delegate.handicappers)
    f.setup()
    return f
reset_opponent_value_func = ResetOpponentValueFunction(save_dir_path, reset_interval, value_func_generator)
callbacks.append(reset_opponent_value_func)

score_output_path = os.path.join(OUTPUT_DIR, "initial_value_transition.csv")
initial_value_scorer = InitialStateValueRecorder(score_output_path)
callbacks.append(initial_value_scorer)

episode_log_path = os.path.join(OUTPUT_DIR, "episode_log.txt")
episode_sample_interval = 1000
episode_sampler = EpisodeSampler(episode_sample_interval, episode_log_path, my_uuid)
callbacks.append(episode_sampler)

weights_output_path = os.path.join(OUTPUT_DIR, "weights_analysis.txt")
weights_sample_interval = 1000
weights_analyzer = WeightsAnalyzer(weights_sample_interval, weights_output_path)
callbacks.append(weights_analyzer)

import pdb; pdb.set_trace()
run_insecure_method(algorithm.run_gpi, (TEST_LENGTH, callbacks))

