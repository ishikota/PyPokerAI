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
import matplotlib
matplotlib.use("tkAgg")
import matplotlib.pyplot as plt
from pypokerai.utils import play_game
from pypokerai.value_function import LinearModelBinaryOnehotFeaturesValueFunction, LinearModelScaledScalarFeaturesValueFunction
from pypokerai.task import blind_structure

import pickle

OUTPUT_DIR_NAME = "game_results"
DATA_PATH = "game_results/"

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), OUTPUT_DIR_NAME)
MAX_ROUND = 1
NB_MATCH = 3000
time_stamp = datetime.now().strftime('%m%d_%H_%M_%S')
with open(DATA_PATH, "rb") as f:
    result_holder = pickle.load(f)

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

