#!/usr/local/bin/python

# Resolve path configucation
import os
import sys

root = os.path.join(os.path.dirname(__file__), "..", "..")
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


from kyoka.algorithm.montecarlo import MonteCarlo, MonteCarloApproxActionValueFunction
from kyoka.algorithm.q_learning import QLearning, QLearningApproxActionValueFunction
from kyoka.algorithm.sarsa import Sarsa, SarsaApproxActionValueFunction
from kyoka.policy import EpsilonGreedyPolicy
from kyoka.callback import LearningRecorder, ManualInterruption

from pypokerai.task import TexasHoldemTask, blind_structure
from pypokerai.value_function import LinearModelScalarFeaturesValueFunction,\
        LinearModelScaledScalarFeaturesValueFunction, LinearModelOnehotFeaturesValueFunction
from pypokerai.callback import ResetOpponentValueFunction


class ApproxActionValueFunction(QLearningApproxActionValueFunction):

    def __init__(self, handicappers=None):
        super(QLearningApproxActionValueFunction, self).__init__()
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

    def save(self, save_dir_path):
        self.delegate.save(save_dir_path)

    def load(self, load_dir_path):
        self.delegate.load(load_dir_path)

TEST_LENGTH = 10000

# Setup algorithm
value_func = ApproxActionValueFunction()
task = TexasHoldemTask()
task.set_opponent_value_functions([value_func]*9)
policy = EpsilonGreedyPolicy(eps=0.99)
policy.set_eps_annealing(0.99, 0.1, TEST_LENGTH)
algorithm = QLearning(gamma=0.99)
algorithm.setup(task, policy, value_func)

# Setup callbacks
save_dir_path = os.path.join(os.path.dirname(__file__), "checkpoint")
learning_recorder = LearningRecorder(algorithm, save_dir_path, 10)

monitor_file_path = os.path.join(os.path.dirname(__file__), "stop.txt")
manual_interruption = ManualInterruption(monitor_file_path)

reset_interval = 15
def value_func_generator():
    f = ApproxActionValueFunction(value_func.delegate.handicappers)
    f.setup()
    return f
reset_opponent_value_func = ResetOpponentValueFunction(save_dir_path, reset_interval, value_func_generator)

callbacks = [learning_recorder, manual_interruption, reset_opponent_value_func]
run_insecure_method(algorithm.run_gpi, (TEST_LENGTH, callbacks))

