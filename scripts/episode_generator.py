#!/usr/local/bin/python

# Resolve path configucation
import os
import sys

root = os.path.join(os.path.dirname(__file__), "../"*5)
src_path = os.path.join(root, "pypokerai")
sys.path.append(root)
sys.path.append(src_path)

from kyoka.algorithm.rl_algorithm import generate_episode
from kyoka.algorithm.montecarlo import MonteCarloApproxActionValueFunction
from kyoka.policy import GreedyPolicy
from pypokerai.task import TexasHoldemTask, blind_structure
from pypokerai.value_function import LinearModelOnehotFeaturesValueFunction
from pypokerai.callback import EpisodeSampler

# CONST
POKER_ROUND = None
if not POKER_ROUND:
    raise Exception("You forget to set max round for n-round poker task.")

# generate handicappers
tmp = LinearModelOnehotFeaturesValueFunction(blind_structure)
tmp.setup()
handicappers = tmp.handicappers

# setup task, policy, value_function
LATEST_DIR = "gpi_finished"
CHECKPOINT_PATH = os.path.join(os.path.dirname(__file__), "checkpoint")
agent_func_load_path = os.path.join(CHECKPOINT_PATH, LATEST_DIR)
opponent_func_load_paths = [
        os.path.join(CHECKPOINT_PATH, LATEST_DIR),
        os.path.join(CHECKPOINT_PATH, LATEST_DIR),
        os.path.join(CHECKPOINT_PATH, LATEST_DIR),
        os.path.join(CHECKPOINT_PATH, LATEST_DIR),
        os.path.join(CHECKPOINT_PATH, LATEST_DIR),
        os.path.join(CHECKPOINT_PATH, LATEST_DIR),
        os.path.join(CHECKPOINT_PATH, LATEST_DIR),
        os.path.join(CHECKPOINT_PATH, LATEST_DIR),
        os.path.join(CHECKPOINT_PATH, LATEST_DIR)
]

class ValueFuncWrapper(MonteCarloApproxActionValueFunction):

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


agent_value_func = ValueFuncWrapper()
agent_value_func.setup()
agent_value_func.load(agent_func_load_path)

opponent_value_funcs = []
for path in opponent_func_load_paths:
    value_func = LinearModelOnehotFeaturesValueFunction(blind_structure, handicappers)
    value_func.setup()
    value_func.load(path)
    opponent_value_funcs.append(value_func)

task = TexasHoldemTask(final_round=POKER_ROUND, scale_reward=True, lose_penalty=True)
task.set_opponent_value_functions(opponent_value_funcs)
greedy_policy = GreedyPolicy()

# generate episode
while True:
    quiet_helper = EpisodeSampler("dummy", "dummy", "dummy", show_weights=False)
    loud_helper = EpisodeSampler("dummy", "dummy", "dummy", show_weights=True)
    episode = generate_episode(task, greedy_policy, agent_value_func)
    print "final reward = %s, episode_length=%d." % (episode[-1][3], len(episode))

    if "y" == raw_input(">> Do you see this episode in detail? (y/n)"):
        for experience in episode:
            print quiet_helper._visualize_action_log(task, agent_value_func, experience)
            if "y" == raw_input("do you want to see weights? (y/n)"):
                print loud_helper._visualize_action_log(task, agent_value_func, experience)
            raw_input(">>> type something to go next...")

    if "y" == raw_input(">> type 'y' to quit the script else generate another episode."):
        break

