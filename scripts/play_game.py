#!/usr/local/bin/python

# Resolve path configucation
import os
import sys

root = os.path.join(os.path.dirname(__file__), "../"*5)
src_path = os.path.join(root, "pypokerai")
sys.path.append(root)
sys.path.append(src_path)

import random
from pypokerai.utils import play_game
from pypokerai.value_function import LinearModelOnehotFeaturesValueFunction
from pypokerai.task import blind_structure

#CHOICE_ALGORITHM = "latest"
CHOICE_ALGORITHM = "random"
LATEST_DIR = "gpi_finished"
CHECKPOINT_PATH = os.path.join(os.path.dirname(__file__), "checkpoint")

# generate handicappers
tmp = LinearModelOnehotFeaturesValueFunction(blind_structure)
tmp.setup()
handicappers = tmp.handicappers

# generate value functions
names = []
value_funcs = []
if CHOICE_ALGORITHM == "latest":
    for i in range(10):
        f = LinearModelOnehotFeaturesValueFunction(blind_structure, handicappers)
        f.setup()
        f.load(os.path.join(CHECKPOINT_PATH, LATEST_DIR))
        names.append("latest-%d" % i)
        value_funcs.append(f)
else:
    load_targets = os.listdir(CHECKPOINT_PATH)
    idxs = [random.randint(0, len(load_targets)-1) for _ in range(10)]
    fnames = [load_targets[i] for i in idxs]
    paths = [os.path.join(CHECKPOINT_PATH, fname) for fname in fnames]
    for name, path in zip(fnames, paths):
        f = LinearModelOnehotFeaturesValueFunction(blind_structure, handicappers)
        f.setup()
        f.load(path)
        names.append(name)
        value_funcs.append(f)

play_game(1, names, value_funcs, with_me=False)

