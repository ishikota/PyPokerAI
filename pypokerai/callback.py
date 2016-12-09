import os
import re
import random

from kyoka.callback import BaseCallback

class ResetOpponentValueFunction(BaseCallback):

    PLAYER_NUM = 9

    def __init__(self, checkpoint_dir_path, reset_interval, value_func_generator, reset_policy="latest"):
        self.checkpoint_dir_path = checkpoint_dir_path
        self.reset_interval = reset_interval
        self.generator_method = value_func_generator
        self.num_matcher = re.compile("after_(\d+)_iteration")
        self.reset_policy = reset_policy
        if reset_policy not in ["latest", "random"]:
            raise ValueError("Unexpected reset policy [ %s ] receigvegd" % reset_policy)

    def before_gpi_start(self, task, value_function):
        self.log("Opponent value functions will be reset every %d iteration with resource from [ %s ] in '%s' policy." %
                (self.reset_interval, self.checkpoint_dir_path, self.reset_policy))

    def before_update(self, iteration_count, task, value_function):
        pass

    def after_update(self, iteration_count, task, _value_function):
        if iteration_count % self.reset_interval == 0:
            load_dirs = self._fetch_load_dirs(self.checkpoint_dir_path)
            value_funcs = [self._setup_value_function(path) for path in load_dirs]
            task.set_opponent_value_functions(value_funcs)
            base_msg = "Reset opponent value function with resource [ %s ]"
            if self.reset_policy == "latest":
                self.log(base_msg % load_dirs[0])
            elif self.reset_policy == "random":
                for load_dir in load_dirs: self.log(base_msg % load_dir)

    def after_gpi_finish(self, task, value_function):
        pass

    def _fetch_load_dirs(self, checkpoint_dir_path):
        dirs = os.listdir(checkpoint_dir_path)
        targets = [path for path in dirs if path.startswith("after")]
        if len(targets) == 0:
            raise Exception("Failed to load value functions from [ %s ]." % checkpoint_dir_path)
        if self.reset_policy == "latest":
            return self._use_latest_item(targets)
        elif self.reset_policy == "random":
            return self._use_random_choiced_item(targets)

    def _fetch_num(self, file_path):
        return int(self.num_matcher.search(file_path).groups()[0])

    def _use_latest_item(self, load_targets):
        iteration_counts = [self._fetch_num(path) for path in load_targets]
        _max_count, fname = max(sorted(zip(iteration_counts, load_targets))[::-1])
        path = os.path.join(self.checkpoint_dir_path, fname)
        return [path]*self.PLAYER_NUM

    def _use_random_choiced_item(self, load_targets):
        idxs = [random.randint(0, len(load_targets)-1) for _ in range(self.PLAYER_NUM)]
        fnames = [load_targets[i] for i in idxs]
        paths = [os.path.join(self.checkpoint_dir_path, fname) for fname in fnames]
        return paths

    def _setup_value_function(self, load_dir_path):
        value_func = self.generator_method()
        value_func.load(load_dir_path)
        return value_func

