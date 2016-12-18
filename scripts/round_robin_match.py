#!/usr/local/bin/python

# Resolve path configucation
import os
import sys

root = os.path.join(os.path.dirname(__file__), "../"*5)
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
from pypokerai.value_function import LinearModelOnehotFeaturesValueFunction
from pypokerai.task import blind_structure

MAX_ROUND = 1
NB_MATCH = 3000
CHOICE_ALGORITHM = "random"
LATEST_DIR = "gpi_finished"
CHECKPOINT_PATH = os.path.join(os.path.dirname(__file__), "checkpoint")

# generate handicappers
tmp = LinearModelOnehotFeaturesValueFunction(blind_structure)
tmp.setup()
handicappers = tmp.handicappers

load_paths = [
        os.path.join(CHECKPOINT_PATH, "after_50000_iteration"),
        os.path.join(CHECKPOINT_PATH, "after_100000_iteration"),
        os.path.join(CHECKPOINT_PATH, "after_150000_iteration"),
        os.path.join(CHECKPOINT_PATH, "after_200000_iteration"),
        os.path.join(CHECKPOINT_PATH, "after_250000_iteration"),
        os.path.join(CHECKPOINT_PATH, "after_300000_iteration"),
        os.path.join(CHECKPOINT_PATH, "after_350000_iteration"),
        os.path.join(CHECKPOINT_PATH, "after_400000_iteration"),
        os.path.join(CHECKPOINT_PATH, "after_450000_iteration"),
        os.path.join(CHECKPOINT_PATH, "after_500000_iteration"),
        ]
names = [
        "after_50000_iteration",
        "after_100000_iteration",
        "after_150000_iteration",
        "after_200000_iteration",
        "after_250000_iteration",
        "after_300000_iteration",
        "after_350000_iteration",
        "after_400000_iteration",
        "after_450000_iteration",
        "after_5000000_iteration",
        ]

value_funcs = []
for name, path in zip(names, load_paths):
    f = LinearModelOnehotFeaturesValueFunction(blind_structure, handicappers)
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

# generate graph
time_stamp = datetime.now().strftime('%m%d_%H_%M_%S')
num_matcher = re.compile("after_(\d+)_iteration")
fetch_num = lambda fname: int(num_matcher.search(fname).groups()[0])
X = range(1, NB_MATCH+1)

# draw all graph on 1 pane
plot_save_path = os.path.join(OUTPUT_DIR, "%d_round_%d_match_plot_%s.png" % (MAX_ROUND, NB_MATCH, time_stamp))
for key, vals in result_holder.items():
    plt.plot(X, vals, label=str(fetch_num(key)))
plt.legend(fontsize=8)
plt.savefig(plot_save_path)
print "1-pane graph saved on [ %s ]" % plot_save_path

# draw each graph on sub pane
def plot_title(title, x=NB_MATCH/2, y=50000):
    plt.text(x, y, title, alpha=0.5, size=10, ha="center", va="center")

sub_plot_save_path = os.path.join(OUTPUT_DIR, "%d_round_%d_match_subplot_%s.png" % (MAX_ROUND, NB_MATCH, time_stamp))
keys = sorted(result_holder.keys(), key=fetch_num)
for idx, key in enumerate(keys):
    vals = result_holder[key]
    avg = 1.0 * sum(vals)/NB_MATCH
    plt.subplot(5, 2, idx+1)
    plot_title("%s (%s)" % (key, avg))
    plt.ylim(0, 100000)
    plt.plot(X, vals)
plt.savefig(sub_plot_save_path)
print "subplot graph saved on [ %s ]" % sub_plot_save_path

