import random
from collections import OrderedDict
from kyoka.task import BaseTask
from kyoka.policy import choose_best_action
from pypokerengine.api.emulator import Emulator
from pypokerengine.players import BasePokerPlayer
from pypokerengine.engine.poker_constants import PokerConstants as Const

# alias
DummyPlayer = BasePokerPlayer

# game settings
max_round = 100000  # no-limit
nb_player = 10
table_break_threshold = 3
initial_stack = 10000
sb_amount = 25
ante = 0
level_interval = 10
blind_structure = {
        level_interval*0 + 1 : { "ante": 0, "small_blind": 25 },  # LEVEL 1
        level_interval*1 + 1 : { "ante": 0, "small_blind": 50 },
        level_interval*2 + 1 : { "ante": 0, "small_blind": 75 },
        level_interval*3 + 1 : { "ante": 0, "small_blind": 100 },
        level_interval*4 + 1 : { "ante": 25, "small_blind": 100 },
        level_interval*5 + 1 : { "ante": 25, "small_blind": 150 },
        level_interval*6 + 1 : { "ante": 50, "small_blind": 200 },
        level_interval*7 + 1 : { "ante": 50, "small_blind": 250 },
        level_interval*8 + 1 : { "ante": 75, "small_blind": 300 },
        level_interval*9 + 1 : { "ante": 100, "small_blind": 400 },
        level_interval*10 + 1 : { "ante": 100, "small_blind": 600 },
        level_interval*11 + 1 : { "ante": 200, "small_blind": 800 },
        level_interval*12 + 1 : { "ante": 200, "small_blind": 1000 },
        level_interval*13 + 1 : { "ante": 300, "small_blind": 1200 },
        level_interval*14 + 1 : { "ante": 400, "small_blind": 1500 },
        level_interval*15 + 1 : { "ante": 500, "small_blind": 2000 },
        level_interval*16 + 1 : { "ante": 500, "small_blind": 2500 },
        level_interval*17 + 1 : { "ante": 500, "small_blind": 3000 },
        level_interval*18 + 1 : { "ante": 1000, "small_blind": 4000 },
        level_interval*19 + 1 : { "ante": 1000, "small_blind": 6000 },
        level_interval*20 + 1 : { "ante": 2000, "small_blind": 8000 },
        level_interval*21 + 1 : { "ante": 3000, "small_blind": 10000 },
        level_interval*22 + 1 : { "ante": 4000, "small_blind": 12000 },
        level_interval*23 + 1 : { "ante": 5000, "small_blind": 15000 },
        level_interval*24 + 1 : { "ante": 10000, "small_blind": 20000 },
        level_interval*25 + 1 : { "ante": 20000, "small_blind": 30000 },  # LEVEL 26
        }
my_uuid = "uuid-0"
my_name = "agent"
players_info = {
        my_uuid : { "stack": initial_stack, "name": my_name },
        "uuid-1": { "stack": initial_stack, "name": "cpu1" },
        "uuid-2": { "stack": initial_stack, "name": "cpu2" },
        "uuid-3": { "stack": initial_stack, "name": "cpu3" },
        "uuid-4": { "stack": initial_stack, "name": "cpu4" },
        "uuid-5": { "stack": initial_stack, "name": "cpu5" },
        "uuid-6": { "stack": initial_stack, "name": "cpu6" },
        "uuid-7": { "stack": initial_stack, "name": "cpu7" },
        "uuid-8": { "stack": initial_stack, "name": "cpu8" },
        "uuid-9": { "stack": initial_stack, "name": "cpu9" }
        }

pick_me = lambda state: [p for p in state["table"].seats.players if p.uuid == my_uuid][0]

class TexasHoldemTask(BaseTask):

    def __init__(self, final_round=max_round, scale_reward=False, lose_penalty=False, shuffle_position=False, action_record=False):
        self.final_round = final_round
        self.scale_reward = scale_reward
        self.lose_penalty = lose_penalty
        self.shuffle_position = shuffle_position
        self.action_record = action_record
        self.emulator = Emulator()
        self.emulator.set_game_rule(nb_player, final_round, sb_amount, ante)
        self.emulator.set_blind_structure(blind_structure)
        self.opponent_value_functions = {}
        if shuffle_position:
            print "Warning: shuffle_position is set True. Are you sure?"

        for uuid in players_info:
            self.emulator.register_player(uuid, DummyPlayer())
            if uuid != my_uuid: self.opponent_value_functions[uuid] = None

    def set_opponent_value_functions(self, value_functions):
        assert len(value_functions) == 9
        opponent_uuids = [uuid for uuid in self.opponent_value_functions if uuid != my_uuid]
        for uuid, value_function in zip(opponent_uuids, value_functions):
            self.opponent_value_functions[uuid] = value_function

    def generate_initial_state(self):
        return self.generate_initial_state_without_action_record() if not self.action_record\
                else self.generate_initial_state_with_action_record()

    def generate_initial_state_without_action_record(self):
        p_info = _get_shuffled_players_info() if self.shuffle_position else players_info
        clear_state = self.emulator.generate_initial_game_state(p_info)
        state, _events = self.emulator.start_new_round(clear_state)
        while not self._check_my_turn(state):
            action, amount = self._choose_opponent_action(state)
            state, _events = self.emulator.apply_action(state, action, amount)
            if state["street"] == Const.Street.FINISHED and not self.is_terminal_state(state):
                state, _events = self.emulator.start_new_round(state)
        return state if not self.is_terminal_state(state) else self.generate_initial_state()

    def generate_initial_state_with_action_record(self):
        p_info = _get_shuffled_players_info() if self.shuffle_position else players_info
        clear_state = self.emulator.generate_initial_game_state(p_info)
        p_act_record = { p.uuid:[[],[],[],[]] for p in clear_state["table"].seats.players }
        state, _events = self.emulator.start_new_round(clear_state)
        while not self._check_my_turn(state):
            state[ACTION_RECORD_KEY] = p_act_record
            opponent_uuid, action_info = self._choose_opponent_action(state, detail_info=True)
            p_act_record = self._update_action_record(state, p_act_record, opponent_uuid, action_info)
            action, amount = action_info["action"], action_info["amount"]
            state, _events = self.emulator.apply_action(state, action, amount)
            if state["street"] == Const.Street.FINISHED and not self.is_terminal_state(state):
                state, _events = self.emulator.start_new_round(state)
        state[ACTION_RECORD_KEY] = p_act_record
        return state if not self.is_terminal_state(state) else self.generate_initial_state()

    def is_terminal_state(self, state):
        me = pick_me(state)
        round_finished = state["street"] == Const.Street.FINISHED
        active_players = [p for p in state["table"].seats.players if p.stack > 0]
        short_of_players = len(active_players) <= table_break_threshold
        i_am_loser = me.stack == 0
        is_final_round = state["round_count"] >= self.final_round
        return round_finished and (short_of_players or i_am_loser or is_final_round)

    def transit_state(self, state, action):
        return self.transit_state_without_action_record(state, action) if not self.action_record\
                else self.transit_state_with_action_record(state, action)

    def transit_state_without_action_record(self, state, action):
        assert self._check_my_turn(state)
        assert not self.is_terminal_state(state)
        action, amount = action["action"], action["amount"]
        state, _events = self.emulator.apply_action(state, action, amount)
        if state["street"] == Const.Street.FINISHED and not self.is_terminal_state(state):
            state, _events = self.emulator.start_new_round(state)
        while not self._check_my_turn(state) and not self.is_terminal_state(state):
            action, amount = self._choose_opponent_action(state)
            state, _events = self.emulator.apply_action(state, action, amount)
            if state["street"] == Const.Street.FINISHED and not self.is_terminal_state(state):
                state, _events = self.emulator.start_new_round(state)
        return state

    def transit_state_with_action_record(self, state, action_info):
        assert self._check_my_turn(state)
        assert not self.is_terminal_state(state)
        assert state.has_key(ACTION_RECORD_KEY)
        p_act_record = _deepcopy_action_record(state)
        p_act_record = self._update_action_record(state, p_act_record, my_uuid, action_info)
        action, amount = action_info["action"], action_info["amount"]
        state, _events = self.emulator.apply_action(state, action, amount)
        state[ACTION_RECORD_KEY] = p_act_record
        if state["street"] == Const.Street.FINISHED and not self.is_terminal_state(state):
            state, _events = self.emulator.start_new_round(state)
        while not self._check_my_turn(state) and not self.is_terminal_state(state):
            state[ACTION_RECORD_KEY] = p_act_record
            opponent_uuid, action_info = self._choose_opponent_action(state, detail_info=True)
            p_act_record = self._update_action_record(state, p_act_record, opponent_uuid, action_info)
            action, amount = action_info["action"], action_info["amount"]
            state, _events = self.emulator.apply_action(state, action, amount)
            if state["street"] == Const.Street.FINISHED and not self.is_terminal_state(state):
                state, _events = self.emulator.start_new_round(state)
        state[ACTION_RECORD_KEY] = p_act_record
        return state

    def _check_my_turn(self, state):
        players = state["table"].seats.players
        return state["next_player"] != "not_found" and my_uuid == players[state["next_player"]].uuid

    def _choose_opponent_action(self, state, detail_info=False):
        players = state["table"].seats.players
        opponent_uuid = players[state["next_player"]].uuid
        value_function = self.opponent_value_functions[opponent_uuid]
        action_info = choose_best_action(self, value_function, state)
        return (opponent_uuid, action_info) if detail_info else (action_info["action"], action_info["amount"])

    def generate_possible_actions(self, state):
        action_info = self.emulator.generate_possible_actions(state)
        min_raise_amount = action_info[2]["amount"]["min"]
        max_raise_amount = action_info[2]["amount"]["max"]
        player = state["table"].seats.players[state["next_player"]]

        actions = [gen_fold_action(), gen_call_action(action_info[1]["amount"])]
        if min_raise_amount != -1:
            actions.append(gen_min_raise_action(min_raise_amount))
        if min_raise_amount != -1 and min_raise_amount*2 < max_raise_amount:
            actions.append(gen_double_raise_action(min_raise_amount*2))
        if min_raise_amount != -1 and min_raise_amount*3 < max_raise_amount:
            actions.append(gen_triple_raise_action(min_raise_amount*3))
        if max_raise_amount != -1:
            actions.append(gen_max_raise_action(max_raise_amount))
        return actions

    def calculate_reward(self, state):
        if self.is_terminal_state(state):
            if pick_me(state).stack == 0 and self.lose_penalty:
                return -1
            if self.scale_reward:
                return 1.0 * pick_me(state).stack / (nb_player * initial_stack)
            else:
                return pick_me(state).stack
        else:
            return 0

    def _update_action_record(self, state, action_record, uuid, action_info):
        action, amount = action_info["action"], action_info["amount"]
        if 'fold' == action: idx = 0
        elif 'call' == action: idx = 1
        elif 'raise' == action: idx = 2
        else: raise Exception("unexpected action [ %s ] received" % action)

        # allin check. the idx of allin is 3.
        action_player = [player for player in state["table"].seats.players if player.uuid==uuid]
        assert len(action_player) == 1
        if amount >= action_player[0].stack and 'fold' != action: idx = 3
        action_record[uuid][idx].append(amount)
        return action_record


def _get_shuffled_players_info():
    shuffled_dict = OrderedDict()
    base_items = players_info.items()
    random.shuffle(base_items)
    for key, val in base_items: shuffled_dict[key] = val
    return shuffled_dict

def _deepcopy_action_record(state):
    original = state[ACTION_RECORD_KEY]
    deepcopy = {}
    for key in original:
        deepcopy[key] = [e[::] for e in original[key]]
    assert original == deepcopy
    return deepcopy

def gen_fold_action():
    return { "name": FOLD, "action": "fold", "amount": 0 }

def gen_call_action(amount):
    return { "name": CALL, "action": "call", "amount": amount }

def gen_min_raise_action(amount):
    return { "name": MIN_RAISE, "action": "raise", "amount": amount }

def gen_double_raise_action(amount):
    return { "name": DOUBLE_RAISE, "action": "raise", "amount": amount }

def gen_triple_raise_action(amount):
    return { "name": TRIPLE_RAISE, "action": "raise", "amount": amount }

def gen_max_raise_action(amount):
    return { "name": MAX_RAISE, "action": "raise", "amount": amount }

# Consts
FOLD = "fold"
CALL = "call"
MIN_RAISE = "min_raise"
DOUBLE_RAISE = "double_raise"
TRIPLE_RAISE = "triple_raise"
MAX_RAISE = "max_raise"
ACTION_RECORD_KEY = "players_action_record"

