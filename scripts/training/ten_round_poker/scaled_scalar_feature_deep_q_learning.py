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

import numpy as np

import shutil
from datetime import datetime

from kyoka.algorithm.montecarlo import MonteCarlo, MonteCarloApproxActionValueFunction
from kyoka.algorithm.q_learning import QLearning, QLearningApproxActionValueFunction
from kyoka.algorithm.sarsa import Sarsa, SarsaApproxActionValueFunction
from kyoka.algorithm.deep_q_learning import DeepQLearning, DeepQLearningApproxActionValueFunction
from kyoka.policy import EpsilonGreedyPolicy
from kyoka.callback import LearningRecorder, ManualInterruption

from pypokerai.task import TexasHoldemTask, blind_structure, my_uuid
from pypokerai.value_function import LinearModelScaledScalarFeaturesValueFunction, action_index
from pypokerai.callback import ResetOpponentValueFunction, InitialStateValueRecorder,\
        EpisodeSampler, WeightsAnalyzer


class ApproxActionValueFunction(DeepQLearningApproxActionValueFunction):

    Q_NET_SAVE_NAME = "q_weight.h5"
    Q_HAT_NET_SAVE_NAME = "q_hat_weight.h5"

    def __init__(self, handicappers=None):
        super(DeepQLearningApproxActionValueFunction, self).__init__()
        self._handicappers = handicappers
        self.prediction_cache = {}  # (features, prediction)
        self.delegate = LinearModelScaledScalarFeaturesValueFunction(blind_structure, self._handicappers)
        self.delegate.setup()

    def initialize_network(self):
        return self.delegate.build_model()

    def deepcopy_network(self, q_network):
        q_hat_network = self.initialize_network()
        for original_layer, copy_layer in zip(q_network.layers, q_hat_network.layers):
            copy_layer.set_weights(original_layer.get_weights())
        return q_hat_network

    def predict_value_by_network(self, network, state, action):
        X, action = self.delegate.construct_features(state, action)
        values = network.predict_on_batch(np.array([X]))[0].tolist()
        valur_for_action = values[action_index(action)]
        return valur_for_action

    def backup_on_minibatch(self, q_network, backup_minibatch):
        X = np.array([self.delegate.construct_features(state, action)[0]
                for state, action, target in backup_minibatch])
        Y_info = [(action, target) for _state, action, target in backup_minibatch]
        Y = q_network.predict_on_batch(X)
        assert len(Y) == len(Y_info)
        for y, (action, target) in zip(Y, Y_info): y[action_index(action)] = target
        loss = q_network.train_on_batch(X, Y)

    def save_networks(self, q_network, q_hat_network, save_dir_path):
        q_network.save_weights(os.path.join(save_dir_path, self.Q_NET_SAVE_NAME))
        q_hat_network.save_weights(os.path.join(save_dir_path, self.Q_HAT_NET_SAVE_NAME))

    def load_networks(self, load_dir_path):
        q_network = self.initialize_network()
        q_network.load_weights(os.path.join(load_dir_path, self.Q_NET_SAVE_NAME))
        q_hat_network = self.initialize_network()
        q_hat_network.load_weights(os.path.join(load_dir_path, self.Q_HAT_NET_SAVE_NAME))
        return q_network, q_hat_network

    def visualize_feature_weights(self):
        return self.delegate.visualize_feature_weights()

# Setup directory to output learning results
time_stamp = datetime.now().strftime('%m%d_%H_%M_%S')
TRAINING_TITLE = "scaled_scalar_feature_deep_q_learning_trash_%s" % time_stamp
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

DQN_PAPER_TEST_LENGTH = 50000000
TEST_LENGTH = 1000000
scale_param = lambda p: int(1.0 * p * TEST_LENGTH / DQN_PAPER_TEST_LENGTH)
N = scale_param(1000000)
C = scale_param(10000)
replay_start_size = scale_param(50000)
assert (N, C, replay_start_size) == (20000, 200, 1000)

# Setup algorithm
value_func = ApproxActionValueFunction()
task = TexasHoldemTask(final_round=10, scale_reward=True, lose_penalty=True)
task.set_opponent_value_functions([value_func]*9)
policy = EpsilonGreedyPolicy(eps=0.99)
policy.set_eps_annealing(0.99, 0.1, int(TEST_LENGTH*0.8))
algorithm = DeepQLearning(gamma=0.99, N=N, C=C, minibatch_size=32, replay_start_size=replay_start_size)

import time
st = time.time()
print "start setup"
algorithm.setup(task, policy, value_func)
print "took time for setup => %s (s)" % (time.time()-st)

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

