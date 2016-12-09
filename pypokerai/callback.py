import os
import re

from kyoka.callback import BaseCallback

class ResetOpponentValueFunction(BaseCallback):

    def __init__(self, checkpoint_dir_path, reset_interval, value_func_generator):
        self.checkpoint_dir_path = checkpoint_dir_path
        self.reset_interval = reset_interval
        self.generator_method = value_func_generator
        self.num_matcher = re.compile("after_(\d+)_iteration")

    def before_gpi_start(self, task, value_function):
        self.log("Opponent value functions will be reset every %d iteration with resource from [ %s ]." %
                (self.reset_interval, self.checkpoint_dir_path))

    def before_update(self, iteration_count, task, value_function):
        pass

    def after_update(self, iteration_count, task, _value_function):
        if iteration_count % self.reset_interval == 0:
            load_dirs = self._fetch_load_dirs(self.checkpoint_dir_path)
            value_funcs = [self._setup_value_function(path) for path in load_dirs]
            task.set_opponent_value_functions(value_funcs)
            self.log("Reset opponent value function with resource [ %s ]" % load_dirs[0])

    def after_gpi_finish(self, task, value_function):
        pass

    def _fetch_load_dirs(self, checkpoint_dir_path):
        dirs = os.listdir(checkpoint_dir_path)
        targets = [path for path in dirs if path.startswith("after")]
        if len(targets) == 0:
            raise Exception("Failed to load value functions from [ %s ]." % checkpoint_dir_path)
        return self._use_latest_item(targets)

    def _fetch_num(self, file_path):
        return int(self.num_matcher.search(file_path).groups()[0])

    def _use_latest_item(self, load_targets):
        iteration_counts = [self._fetch_num(path) for path in load_targets]
        _max_count, fname = max(sorted(zip(iteration_counts, load_targets))[::-1])
        path = os.path.join(self.checkpoint_dir_path, fname)
        return [path]*9

    def _setup_value_function(self, load_dir_path):
        value_func = self.generator_method()
        value_func.load(load_dir_path)
        return value_func

