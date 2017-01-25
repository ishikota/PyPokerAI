from pypokerengine.players import BasePokerPlayer
from pypokerengine.utils.game_state_utils import restore_game_state, attach_hole_card_from_deck
from pypokerengine.utils.card_utils import gen_cards
import pypokerengine.utils.visualize_utils as U
from kyoka.policy import choose_best_action
from pypokerai.features import cards_to_scaled_scalar
from pypokerai.task import ACTION_RECORD_KEY

class PokerPlayer(BasePokerPlayer):

    def __init__(self, task, value_function, debug=True):
        self.task = task
        self.value_func = value_function
        self.debug_mode = debug

    def declare_action(self, valid_actions, hole_card, round_state):
        game_state = restore_state(hole_card, round_state)
        game_state[ACTION_RECORD_KEY] = self.players_action_record
        action = choose_best_action(self.task, self.value_func, game_state)
        self._update_action_record(self.uuid, action["action"], action["amount"], round_state)
        if self.debug_mode:
            win_rate = cards_to_scaled_scalar(
                    round_state, hole_card, "neuralnet",
                    neuralnets=self.value_func.handicappers)
            acts = self.task.generate_possible_actions(game_state)
            act_names = [act["name"] for act in acts]
            act_vals = [self.value_func.predict_value(game_state, a) for a in acts]
            print ""
            print "-- debug info [ %s ] --" % fetch_name(round_state, self.uuid)
            print "hole card=%s (win_rate=%s)" % (hole_card, win_rate)
            print "pot = %s" % round_state["pot"]
            print zip(act_names, act_vals)
            print ""
        return action["action"], action["amount"]

    def receive_game_start_message(self, game_info):
        self.players_action_record = {
                player_info["uuid"]: [[],[],[],[]] for player_info in game_info["seats"]
                }

    def receive_round_start_message(self, round_count, hole_card, seats):
        self.round_count = round_count
        self.hole_card = hole_card

    def receive_street_start_message(self, street, round_state):
        self.street = street

    def receive_game_update_message(self, action, round_state):
        uuid, action, amount = action["player_uuid"], action["action"], action["amount"]
        self._update_action_record(uuid, action, amount, round_state)

    def _update_action_record(self, uuid, action, amount, round_state):
        idx = 100
        if 'fold' == action: idx = 0
        elif 'call' == action: idx = 1
        elif 'raise' == action: idx = 2
        else: raise Exception("unexpected action [ %s ] received" % action)

        # allin check. the idx of allin is 3.
        action_player = [player for player in round_state["seats"] if player["uuid"]==uuid]
        assert len(action_player) == 1
        if 'allin' == action_player[0]["state"] and 'fold' != action: idx = 3
        self.players_action_record[uuid][idx].append(amount)


    def receive_round_result_message(self, winners, hand_info, round_state):
        pass

def restore_state(hole_card, round_state):
    game_state = restore_game_state(round_state)
    for player in game_state["table"].seats.players:
        game_state = attach_hole_card_from_deck(game_state, player.uuid)
    my_hole_card = gen_cards(hole_card)
    me = game_state["table"].seats.players[game_state["next_player"]]
    me.hole_card = my_hole_card
    return game_state

def fetch_name(round_state, uuid):
    return [p["name"] for p in round_state["seats"] if p["uuid"] == uuid][0]

class ConsolePlayer(BasePokerPlayer):

    def declare_action(self, valid_actions, hole_card, round_state):
        print U.visualize_declare_action(valid_actions, hole_card, round_state, self.uuid)
        action, amount = self.__receive_action_from_console(valid_actions)
        return action, amount

    def receive_game_start_message(self, game_info):
        print U.visualize_game_start(game_info, self.uuid)
        self.__wait_until_input()

    def receive_round_start_message(self, round_count, hole_card, seats):
        print U.visualize_round_start(round_count, hole_card, seats, self.uuid)
        self.__wait_until_input()

    def receive_street_start_message(self, street, round_state):
        print U.visualize_street_start(street, round_state, self.uuid)
        self.__wait_until_input()

    def receive_game_update_message(self, new_action, round_state):
        print U.visualize_game_update(new_action, round_state, self.uuid)
        self.__wait_until_input()

    def receive_round_result_message(self, winners, hand_info, round_state):
        print U.visualize_round_result(winners, hand_info, round_state, self.uuid)
        self.__wait_until_input()

    def __wait_until_input(self):
        raw_input("Enter some key to continue ...")

    def __gen_raw_input_wrapper(self):
        return lambda msg: raw_input(msg)

    def __receive_action_from_console(self, valid_actions):
        flg = raw_input('Enter f(fold), c(call), r(raise).\n >> ')
        if flg in self.__gen_valid_flg(valid_actions):
            if flg == 'f':
                return valid_actions[0]['action'], valid_actions[0]['amount']
            elif flg == 'c':
                return valid_actions[1]['action'], valid_actions[1]['amount']
            elif flg == 'r':
                valid_amounts = valid_actions[2]['amount']
                raise_amount = self.__receive_raise_amount_from_console(valid_amounts['min'], valid_amounts['max'])
                return valid_actions[2]['action'], raise_amount
        else:
            return self.__receive_action_from_console(valid_actions)

    def __gen_valid_flg(self, valid_actions):
        flgs = ['f', 'c']
        is_raise_possible = valid_actions[2]['amount']['min'] != -1
        if is_raise_possible:
            flgs.append('r')
        return flgs

    def __receive_raise_amount_from_console(self, min_amount, max_amount):
        raw_amount = self.input_receiver("valid raise range = [%d, %d]" % (min_amount, max_amount))
        try:
            amount = int(raw_amount)
            if min_amount <= amount and amount <= max_amount:
                return amount
            else:
                print "Invalid raise amount %d. Try again."
                return self.__receive_raise_amount_from_console(min_amount, max_amount)
        except:
            print "Invalid input received. Try again."
            return self.__receive_raise_amount_from_console(min_amount, max_amount)

