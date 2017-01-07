import os
import re
import random
import csv
import time

from kyoka.callback import BaseCallback
from kyoka.policy import GreedyPolicy, choose_best_action
from kyoka.algorithm.rl_algorithm import generate_episode
from pypokerengine.utils.visualize_utils import visualize_declare_action
from pypokerengine.engine.data_encoder import DataEncoder
from pypokerengine.engine.action_checker import ActionChecker

from pypokerai.task import blind_structure, FOLD, CALL, MIN_RAISE, DOUBLE_RAISE, TRIPLE_RAISE, MAX_RAISE

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

class InitialStateValueRecorder(BaseCallback):

    def __init__(self, score_file_path):
        self.score_file_path = score_file_path
        self.score_holder = []

    def before_gpi_start(self, task, value_function):
        value = self._predict_value_of_initial_state(task, value_function)
        self.log("Value of initial state is [ %s ]" % value)
        self.score_holder.append(value)

    def after_update(self, iteration_count, task, value_function):
        value = self._predict_value_of_initial_state(task, value_function)
        self.log("Value of initial state is [ %s ]" % value)
        self.score_holder.append(value)

    def after_gpi_finish(self, task, value_function):
        with open(self.score_file_path, "wb") as f:
            writer = csv.writer(f, lineterminator="\n")
            writer.writerow(self.score_holder)
        self.log("Score is saved on [ %s ]" % self.score_file_path)


    def _predict_value_of_initial_state(self, task, value_function):
        state = task.generate_initial_state()
        while task.is_terminal_state(state): state = task.generate_initial_state()
        action = choose_best_action(task, value_function, state)
        return value_function.predict_value(state, action)

class TrainingLossRecorder(BaseCallback):

    def __init__(self, record_file_path):
        self.record_file_path = record_file_path

    def before_gpi_start(self, task, value_function):
        self.log("Record training loss")
        value_function.delegate.loss_history  # validation for old code

    def after_update(self, iteration_count, task, value_function):
        last_training_loss = value_function.delegate.loss_history[-1]
        self.log("Training loss is [ %s ]" % last_training_loss)

    def after_gpi_finish(self, task, value_function):
        with open(self.record_file_path, "wb") as f:
            writer = csv.writer(f, lineterminator="\n")
            writer.writerow(value_function.delegate.loss_history)
        self.log("Loss history is saved on [ %s ]" % self.record_file_path)

class EpisodeSampler(BaseCallback):

    def __init__(self, sample_interval, log_file_path, my_uuid, show_weights=False):
        self.sample_interval = sample_interval
        self.log_fpath = log_file_path
        self.greedy_policy = GreedyPolicy()
        self.my_uuid = my_uuid
        self.show_weights = show_weights

    def before_gpi_start(self, tack, value_function):
        self.log("Sample episode after each %d iteration and log it on [ %s ]"
                % (self.sample_interval, self.log_fpath))

    def after_update(self, iteration_count, task, value_function):
        if iteration_count % self.sample_interval == 0:
            st = time.time()
            episode = generate_episode(task, self.greedy_policy, value_function)
            calc_time = time.time() - st
            round_count = episode[-1][2]["round_count"]
            final_reward = episode[-1][3]
            self.log("Episode finished at %d round with reward = %s (took %s sec)" %
                    (round_count, final_reward, calc_time))
            self.write_action_log_to_file(iteration_count, task, value_function, episode, self.my_uuid, final_reward)

    def write_action_log_to_file(self, iteration_count, task, value_function, episode, my_uuid, final_reward):
        header_divider = "*"*40
        header_content = "After %d iteration (final reward = %s)" % (iteration_count, final_reward)
        header = "\n".join([header_divider, header_content, header_divider])
        action_logs = [self._visualize_action_log(task, value_function, e) for e in episode]
        action_logs = action_logs
        logs = header + "\n" + "\n\n".join(action_logs) + "\n\n\n"
        with open(self.log_fpath, "a") as f: f.write(logs)

    def _visualize_action_log(self, task, value_function, experience):
        state, action, _next_state, _reward = experience
        players = state["table"].seats.players
        me = [p for p in players if p.uuid == "uuid-0"][0]
        me_pos = players.index(me)
        sb_amount = state["small_blind_amount"]
        valid_actions = ActionChecker.legal_actions(players, me_pos, sb_amount)
        hole = [str(card) for card in me.hole_card]
        round_state = DataEncoder.encode_round_state(state)
        visualized_state = visualize_declare_action(valid_actions, hole, round_state)
        action_log = "Agent took action [ %s: %s (%s) ] at round %d" % (
                action["action"], action["amount"], action["name"], state["round_count"])
        actions = task.generate_possible_actions(state)
        act_vals = [value_function.predict_value(state, act) for act in actions]
        act_names = [act["name"] for act in actions]
        action_value_log = "  => %s" % zip(act_names, act_vals)
        if self.show_weights:
            weights_log = ["** weights and features in detail **"]
            features_title = value_function.delegate.generate_features_title()
            features = value_function.delegate.construct_poker_features(
                    "dummy", "dummy", round_state, me.uuid, hole,
                    value_function.delegate.handicappers, blind_structure)
            w_for_acts = value_function.delegate.model.get_weights()[0].T
            bias = value_function.delegate.model.layers[0].b.get_value()
            weights_log.append("features : %s" % features)
            weights_log.append("bias : %s" % bias)
            for act in actions:
                weights_log.append("")
                act_val = act_vals[actions.index(act)]
                weights = w_for_acts[actions.index(act)].tolist()
                linear_comb = [(f*w, f, w, features_title[idx]) for idx,(f,w) in enumerate(zip(features, weights))]
                linear_comb.append((bias[actions.index(act)], 0, 0, "bias"))
                linear_comb = sorted(linear_comb, key=lambda item: abs(item[0]))[::-1]

                display_items = []
                for item in linear_comb:
                    display_items.append(item)
                    if abs(act_val - sum([tpl[0] for tpl in display_items])) < 0.01:
                        break

                weights_log.append("linear combination for %s :" % act["name"])
                for comb in display_items:
                    weights_log.append("    %s" % str(comb))
            action_value_log += "\n" + "\n".join(weights_log)
        return "\n".join([visualized_state, action_log, action_value_log])

class WeightsAnalyzer(BaseCallback):

    def __init__(self, sample_interval, log_file_path):
        self.sample_interval = sample_interval
        self.log_fpath = log_file_path

    def before_gpi_start(self, tack, value_function):
        self.log("Analyze model weights after each %d iteration and log it on [ %s ]"
                % (self.sample_interval, self.log_fpath))
        header_content = "Initial weights"
        self.write_visualized_weights_to_file(value_function, header_content)

    def after_update(self, iteration_count, task, value_function):
        if iteration_count % self.sample_interval == 0:
            header_content = "After %d iteration" % iteration_count
            self.write_visualized_weights_to_file(value_function, header_content)
            self.log("Logged weights analysis")

    def write_visualized_weights_to_file(self, value_function, header_content):
            header_divider = "*"*40
            header = "\n".join([header_divider, header_content, header_divider])
            content = value_function.visualize_feature_weights()
            logs = header + "\n" + content + "\n\n\n"
            with open(self.log_fpath, "a") as f: f.write(logs)

