#!/usr/local/bin/python

# Resolve path configucation
import os
import sys

root = os.path.join(os.path.dirname(__file__), "../"*3)
src_path = os.path.join(root, "pypokerai")
sys.path.append(root)
sys.path.append(src_path)

from datetime import datetime
import time
import re
import random
import matplotlib
matplotlib.use("tkAgg")
import matplotlib.pyplot as plt
from pypokerai.utils import play_game
from pypokerai.value_function import LinearModelOnehotFeaturesValueFunction, LinearModelBinaryOnehotFeaturesValueFunction,\
        LinearModelScalarFeaturesValueFunction, LinearModelScaledScalarFeaturesValueFunction, RandomValueFunction
from pypokerai.task import blind_structure

MAX_ROUND = None
NB_MATCH = 3000
LATEST_DIR = "gpi_finished"
CHECKPOINT_PATH = os.path.join(os.path.dirname(__file__), "checkpoint")

# validatioin
if not MAX_ROUND:
    raise Exception("You forget to set max round for n-round poker task.")

# generate handicappers
tmp = LinearModelOnehotFeaturesValueFunction(blind_structure)
tmp.setup()
handicappers = tmp.handicappers

constructors = [
        LinearModelScalarFeaturesValueFunction,
        LinearModelScalarFeaturesValueFunction,
        LinearModelScaledScalarFeaturesValueFunction,
        LinearModelScaledScalarFeaturesValueFunction,
        LinearModelOnehotFeaturesValueFunction,
        LinearModelOnehotFeaturesValueFunction,
        LinearModelBinaryOnehotFeaturesValueFunction,
        LinearModelBinaryOnehotFeaturesValueFunction,
        RandomValueFunction,
        RandomValueFunction
        ]

load_paths = [
        "results/scalar_feature_montecarlo_trash_1224_02_47_16/checkpoint/after_1000000_iteration",
        "results/scalar_feature_montecarlo_trash_1224_02_47_16/checkpoint/gpi_finished",
        "results/scaled_scalar_feature_montecarlo_trash_1224_03_05_58/checkpoint/after_1000000_iteration",
        "results/scaled_scalar_feature_montecarlo_trash_1224_03_05_58/checkpoint/gpi_finished",
        "results/montecarlo_after_500000_training_1217_23_37_17/checkpoint/gpi_finished",
        "results/montecarlo_after_500000_training_1217_23_37_17/checkpoint/gpi_finished",
        "results/binary_feature_montecarlo_trash_1222_17_26_05/checkpoint/after_1000000_iteration",
        "results/binary_feature_montecarlo_trash_1222_17_26_05/checkpoint/gpi_finished",
        "dummy",
        "dummy",
        ]
names = [
        "scalar_features_1",
        "scalar_features_2",
        "scaled_scalar_features_1",
        "scaled_scalar_features_2",
        "onehot_features_1",
        "onehot_features_2",
        "binary_onehot_features_1",
        "binary_onehot_features_2",
        "random_player_1",
        "random_player_2",
        ]

value_funcs = []
for constructor, name, path in zip(constructors, names, load_paths):
    f = constructor(blind_structure, handicappers)
    f.setup()
    f.load(path)
    value_funcs.append(f)

result_holder = { name: [] for name in names }
assert len(result_holder.keys()) == 10

start_time = time.time()
for i in range(NB_MATCH):
    zipped = zip(names, value_funcs)
    random.shuffle(zipped)
    names = [name for name, _f in zipped]
    value_funcs = [func for _n, func in zipped]
    result = play_game(MAX_ROUND, names, value_funcs, with_me=False, verbose=0)
    for player_info in result["players"]:
        result_holder[player_info["name"]].append(player_info["stack"])
    print "%d / %d finished" % (i+1, NB_MATCH)
print "round-robin %d times took %s sec" % (NB_MATCH, time.time()-start_time)

# Create output directory if needed
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "game_results")
if not os.path.exists(OUTPUT_DIR):
    os.mkdir(OUTPUT_DIR)

# generate graph from here
time_stamp = datetime.now().strftime('%m%d_%H_%M_%S')
num_matcher = re.compile("after_(\d+)_iteration")
fetch_num = lambda fname: int(num_matcher.search(fname).groups()[0])
X = range(1, NB_MATCH+1)
original_font_size = plt.rcParams["font.size"]

# draw each graph on sub pane
plt.rcParams["font.size"] = 7
sub_plot_save_path = os.path.join(OUTPUT_DIR, "%d_round_%d_match_subplot_%s.png" % (MAX_ROUND, NB_MATCH, time_stamp))
keys = sorted(result_holder.keys())
fig = plt.figure()
for i, key in enumerate(keys):
    vals = result_holder[key]
    avg = 1.0 * sum(vals)/NB_MATCH
    ax = fig.add_subplot(5, 2, i+1)
    ax.title.set_text("%s (avg=%.2f)" % (key, avg))
    ax.set_ylim(0, 100000)
    ax.plot(X, vals)
fig.tight_layout()
plt.savefig(sub_plot_save_path)
plt.rcParams["font.size"] = original_font_size
print "subplot graph saved on [ %s ]" % sub_plot_save_path

# draw histgram
hist_save_path = os.path.join(OUTPUT_DIR, "%d_round_%d_match_histgram_%s.png" % (MAX_ROUND, NB_MATCH, time_stamp))
plt.rcParams["font.size"] = 7
keys = sorted(result_holder.keys())
fig = plt.figure()
for i, key in enumerate(keys):
    vals = result_holder[key]
    ax = fig.add_subplot(5, 2, i+1)
    ax.title.set_text(key)
    ax.hist(vals, bins=30)
fig.tight_layout()
plt.savefig(hist_save_path)
plt.rcParams["font.size"] = original_font_size
print "histgram saved on [ %s ]" % hist_save_path

