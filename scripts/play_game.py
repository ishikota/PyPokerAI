#!/usr/local/bin/python

# Resolve path configucation
import os
import sys

root = os.path.join(os.path.dirname(__file__), "..", "..")
src_path = os.path.join(root, "pypokerai")
sys.path.append(root)
sys.path.append(src_path)

from pypokerai.utils import play_game
from pypokerai.value_function import LinearModelOnehotFeaturesValueFunction
from pypokerai.task import blind_structure

VALUE_FUNC_RESOURCE_PATH = os.path.join(os.path.dirname(__file__), "training/one_round_poker/results/100000_episode_1213/checkpoint/gpi_finished")

value_func = LinearModelOnehotFeaturesValueFunction(blind_structure)
value_func.setup()
value_func.load(VALUE_FUNC_RESOURCE_PATH)
value_funcs = [value_func]*10
play_game(value_funcs, with_me=False)

