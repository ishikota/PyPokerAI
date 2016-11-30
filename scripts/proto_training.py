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

from kyoka.algorithm.montecarlo import MonteCarlo,\
        MonteCarloTabularActionValueFunction, MonteCarloApproxActionValueFunction
from kyoka.policy import EpsilonGreedyPolicy

from pypokerengine.engine.data_encoder import DataEncoder

class ApproxActionValueFunction(MonteCarloApproxActionValueFunction):

    def construct_features(self, state, action):
        my_uuid = state["table"].seats.players[state["next_player"]].uuid
        hole_card = [p for p in state["table"].seats.players if p.uuid==my_uuid][0].hole_card
        hole_str = [str(card) for card in hole_card]
        round_state = DataEncoder.encode_round_state(state)
        features = construct_scaled_scalar_features(round_state, my_uuid, hole_str, blind_structure)
        return state

    def approx_predict_value(self, features):
        return 0

    def approx_backup(self, features, backup_target, alpha):
        pass

TEST_LENGTH = 10000
value_func = ApproxActionValueFunction()
task = TexasHoldemTask()
task.value_function = value_func
policy = EpsilonGreedyPolicy(eps=0.1)
algorithm = MonteCarlo(gamma=0.01)
algorithm.setup(task, policy, value_func)
import pdb, traceback, sys
try:
    algorithm.run_gpi(TEST_LENGTH)
except:
    type, value, tb = sys.exc_info()
    traceback.print_exc()
    pdb.post_mortem(tb)
