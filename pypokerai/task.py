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
blind_structure = {}  # TODO AJPC 2016 main tornament structure
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

    def __init__(self):
        self.emulator = Emulator()
        self.emulator.set_game_rule(nb_player, max_round, sb_amount, ante)
        self.emulator.set_blind_structure(blind_structure)
        for uuid in players_info:
            self.emulator.register_player(uuid, DummyPlayer())

    def generate_initial_state(self):
        clear_state = self.emulator.generate_initial_game_state(players_info)
        state, _events = self.emulator.start_new_round(clear_state)
        while not self._check_my_turn(state):
            action, amount = self._choose_opponent_action(state, self.value_function)
            state, _events = self.emulator.apply_action(state, action, amount)
        return state

    def is_terminal_state(self, state):
        me = pick_me(state)
        round_finished = state["street"] == Const.Street.FINISHED
        active_players = [p for p in state["table"].seats.players if p.stack > 0]
        short_of_players = len(active_players) <= table_break_threshold
        i_am_loser = me.stack == 0
        return round_finished and (short_of_players or i_am_loser)

    def transit_state(self, state, action):
        assert self._check_my_turn(state)
        action, amount = action["action"], action["amount"]
        state, _events = self.emulator.apply_action(state, action, amount)
        if state["street"] == Const.Street.FINISHED:
            state, _events = self.emulator.start_new_round(state)
        while not self._check_my_turn(state) and not self.is_terminal_state(state):
            action, amount = self._choose_opponent_action(state, self.value_function)
            state, _events = self.emulator.apply_action(state, action, amount)
            if state["street"] == Const.Street.FINISHED:
                state, _events = self.emulator.start_new_round(state)
        return state

    def _check_my_turn(self, state):
        players = state["table"].seats.players
        return state["next_player"] != "not_found" and my_uuid == players[state["next_player"]].uuid

    def _choose_opponent_action(self, state, value_function):
        action_info = choose_best_action(self, value_function, state)
        return action_info["action"], action_info["amount"]

    def generate_possible_actions(self, state):
        action_info = self.emulator.generate_possible_actions(state)
        min_raise_amount = action_info[2]["amount"]["min"]
        max_raise_amount = action_info[2]["amount"]["max"]
        player = state["table"].seats.players[state["next_player"]]

        actions = [gen_fold_action(), gen_call_action(action_info[1]["amount"])]
        if min_raise_amount != -1:
            actions.append(gen_min_raise_action(min_raise_amount))
        if min_raise_amount*2 < max_raise_amount:
            actions.append(gen_double_raise_action(min_raise_amount*2))
        if min_raise_amount*3 < max_raise_amount:
            actions.append(gen_triple_raise_action(min_raise_amount*3))
        if max_raise_amount != -1:
            actions.append(gen_max_raise_action(max_raise_amount))
        return actions

    def calculate_reward(self, state):
        if self.is_terminal_state(state):
            return pick_me(state).stack
        else:
            return 0

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

