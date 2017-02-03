#!/usr/local/bin/python

# Resolve path configucation
import os
import sys

root = os.path.join(os.path.dirname(__file__), "../"*5)
src_path = os.path.join(root, "pypokerai")
sys.path.append(root)
sys.path.append(src_path)

from datetime import datetime
import pickle
import time
import re
import random
#import matplotlib
#matplotlib.use("tkAgg")
#import matplotlib.pyplot as plt
from pypokerai.utils import play_game
from pypokerai.value_function import MLPTwoLayerScaledScalarFeaturesValueFunction,\
        MLPFiveLayerScaledScalarFeaturesValueFunction
from pypokerai.task import blind_structure

VALUE_FUNC_CLASS = None
MAX_ROUND = None
NB_MATCH = 3000
LATEST_DIR = "gpi_finished"
CHECKPOINT_PATH = os.path.join(os.path.dirname(__file__), "checkpoint")

# validatioin
if not MAX_ROUND:
    raise Exception("You forget to set max round for n-round poker task.")

if not VALUE_FUNC_CLASS:
    raise Exception("You forget to set value function type")

# generate handicappers
tmp = VALUE_FUNC_CLASS(blind_structure)
tmp.setup()
handicappers = tmp.handicappers

load_paths = [
        os.path.join(CHECKPOINT_PATH, "after_100000_iteration"),
        os.path.join(CHECKPOINT_PATH, "after_200000_iteration"),
        os.path.join(CHECKPOINT_PATH, "after_300000_iteration"),
        os.path.join(CHECKPOINT_PATH, "after_400000_iteration"),
        os.path.join(CHECKPOINT_PATH, "after_500000_iteration"),
        os.path.join(CHECKPOINT_PATH, "after_600000_iteration"),
        os.path.join(CHECKPOINT_PATH, "after_700000_iteration"),
        os.path.join(CHECKPOINT_PATH, "after_800000_iteration"),
        os.path.join(CHECKPOINT_PATH, "after_900000_iteration"),
        os.path.join(CHECKPOINT_PATH, "after_1000000_iteration"),
        ]
names = [
        "after_100000_iteration",
        "after_200000_iteration",
        "after_300000_iteration",
        "after_400000_iteration",
        "after_500000_iteration",
        "after_600000_iteration",
        "after_700000_iteration",
        "after_800000_iteration",
        "after_900000_iteration",
        "after_1000000_iteration",
        ]

value_funcs = []
for name, path in zip(names, load_paths):
    f = VALUE_FUNC_CLASS(blind_structure, handicappers)
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

time_stamp = datetime.now().strftime('%m%d_%H_%M_%S')

# save round-robin result by pickle
data_save_path = os.path.join(OUTPUT_DIR, "%d_round_%d_match_data_%s.pickle" % (MAX_ROUND, NB_MATCH, time_stamp))
with open(data_save_path, "wb") as f:
    pickle.dump(result_holder, f)
print "result data is saved on [ %s ]" % data_save_path

"""
# generate graph from here
num_matcher = re.compile("after_(\d+)_iteration")
fetch_num = lambda fname: int(num_matcher.search(fname).groups()[0])
X = range(1, NB_MATCH+1)
original_font_size = plt.rcParams["font.size"]

# draw each graph on sub pane
plt.rcParams["font.size"] = 7
sub_plot_save_path = os.path.join(OUTPUT_DIR, "%d_round_%d_match_subplot_%s.png" % (MAX_ROUND, NB_MATCH, time_stamp))
keys = sorted(result_holder.keys(), key=fetch_num)
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
keys = sorted(result_holder.keys(), key=fetch_num)
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
"""
