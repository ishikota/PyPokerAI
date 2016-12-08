#!/usr/local/bin/python

# Resolve path configucation
import os
import sys

root = os.path.join(os.path.dirname(__file__), "..")
src_path = os.path.join(root, "pypokerai")
sys.path.append(root)
sys.path.append(src_path)

from pypokerai.task import TexasHoldemTask, blind_structure
from pypokerai.features import construct_scalar_features, construct_scaled_scalar_features, construct_onehot_features
from pypokerai.value_function import LinearModelScalarFeaturesValueFunction,\
        LinearModelScaledScalarFeaturesValueFunction, LinearModelOnehotFeaturesValueFunction

from kyoka.algorithm.montecarlo import MonteCarlo, MonteCarloApproxActionValueFunction
from kyoka.algorithm.q_learning import QLearning, QLearningApproxActionValueFunction
from kyoka.algorithm.sarsa import Sarsa, SarsaApproxActionValueFunction
from kyoka.policy import EpsilonGreedyPolicy


class ApproxActionValueFunction(QLearningApproxActionValueFunction):

    def setup(self):
        self.delegate = LinearModelOnehotFeaturesValueFunction(blind_structure)
        self.delegate.setup()

    def construct_features(self, state, action):
        return self.delegate.construct_features(state, action)

    def approx_predict_value(self, features):
        return self.delegate.approx_predict_value(features)

    def approx_backup(self, features, backup_target, alpha):
        self.delegate.approx_backup(features, backup_target, alpha)

TEST_LENGTH = 10000
value_func = ApproxActionValueFunction()
task = TexasHoldemTask()
task.set_value_function(value_func)
policy = EpsilonGreedyPolicy(eps=0.1)
algorithm = QLearning(gamma=0.99)
algorithm.setup(task, policy, value_func)
import pdb, traceback, sys
try:
    algorithm.run_gpi(TEST_LENGTH)
except:
    type, value, tb = sys.exc_info()
    traceback.print_exc()
    pdb.post_mortem(tb)
