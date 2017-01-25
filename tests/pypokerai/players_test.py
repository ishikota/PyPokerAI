from tests.base_unittest import BaseUnitTest
from mock import Mock

from pypokerai.task import TexasHoldemTask, blind_structure
from pypokerai.player import PokerPlayer
from pypokerai.value_function import LinearModelScaledScalarFeaturesWithActionRecordValueFunction

class PlayersTest(BaseUnitTest):

    def setUp(self):
        self.task = TexasHoldemTask()
        value_func = Mock()
        value_func.predict_value.side_effect = lambda state, action: 1 if action["action"]=="call" else 0
        value_func.setup()
        self.player = PokerPlayer(self.task, value_func, debug=False)

    def test_receive_game_start_message(self):
        self.player.receive_game_start_message(game_info)
        self.size(3, self.player.players_action_record)
        self.eq([[]]*4, self.player.players_action_record["vzrsrzvmyzaipkcenzdqwp"])
        self.eq([[]]*4, self.player.players_action_record["bwivgajwiewdkpymuztbxq"])
        self.eq([[]]*4, self.player.players_action_record["pdvfqlaedvtdxezzhqzmdb"])

    def test_receive_round_start_message(self):
        round_count = 1
        hole_card = ['H8', 'H7']
        seats = [
                {'stack': 90, 'state': 'participating', 'name': 'p1', 'uuid': 'vzrsrzvmyzaipkcenzdqwp'},
                {'stack': 85, 'state': 'participating', 'name': 'p2', 'uuid': 'bwivgajwiewdkpymuztbxq'},
                {'stack': 80, 'state': 'participating', 'name': 'p3', 'uuid': 'pdvfqlaedvtdxezzhqzmdb'}
                ]
        self.player.receive_round_start_message(round_count, hole_card, seats)
        self.eq(round_count, self.player.round_count)
        self.eq(hole_card, self.player.hole_card)

    def test_receive_street_start_message(self):
        street = 'preflop'
        round_state = "dummy"
        self.player.receive_street_start_message(street, round_state)
        self.eq(street, self.player.street)

    def test_receive_game_update_message(self):
        self.player.receive_game_start_message(game_info)
        self.player.receive_round_start_message(1, ['H8', 'H7'], "dummy")
        action_call = {'player_uuid': 'vzrsrzvmyzaipkcenzdqwp', 'action': 'call', 'amount': 10}
        self.player.receive_game_update_message(action_call, round_state_call)
        action_raise = {'player_uuid': 'bwivgajwiewdkpymuztbxq', 'action': 'raise', 'amount': 15}
        self.player.receive_game_update_message(action_raise, round_state_raise)
        action_allin_raise = {'player_uuid': 'pdvfqlaedvtdxezzhqzmdb', 'action': 'raise', 'amount': 90}
        self.player.receive_game_update_message(action_allin_raise, round_state_allin_raise)
        action_allin_call = {'player_uuid': 'vzrsrzvmyzaipkcenzdqwp', 'action': 'call', 'amount': 90}
        self.player.receive_game_update_message(action_allin_call, round_state_allin_call)
        action_fold = {'player_uuid': 'bwivgajwiewdkpymuztbxq', 'action': 'fold', 'amount': 0}
        self.player.receive_game_update_message(action_fold, round_state_fold)
        self.eq([[],[10],[],[90]], self.player.players_action_record["vzrsrzvmyzaipkcenzdqwp"])
        self.eq([[0],[],[15],[]], self.player.players_action_record["bwivgajwiewdkpymuztbxq"])
        self.eq([[],[],[],[90]], self.player.players_action_record["pdvfqlaedvtdxezzhqzmdb"])

    def test_declare_action_test(self):
        valid_actions = "dummy"
        hole_card = ['H8', 'H7']
        round_state = round_state_call
        self.player.receive_game_start_message(game_info)
        self.player.receive_round_start_message(1, ['H8', 'H7'], "dummy")
        self.player.set_uuid("vzrsrzvmyzaipkcenzdqwp")
        action, amount = self.player.declare_action(valid_actions, hole_card, round_state)
        assert "call" == action and 10 == amount
        act_record = self.player.value_func.predict_value.call_args_list[0][0][0]["players_action_record"]
        self.eq([10], act_record[self.player.uuid][1])

game_info = {
        'player_num': 3,
        'rule': {
            'ante': 10,
            'blind_structure': {},
            'max_round': 10,
            'initial_stack': 100,
            'small_blind_amount': 5
            },
        'seats': [
            {'stack': 100, 'state': 'participating', 'name': 'p1', 'uuid': 'vzrsrzvmyzaipkcenzdqwp'},
            {'stack': 100, 'state': 'participating', 'name': 'p2', 'uuid': 'bwivgajwiewdkpymuztbxq'},
            {'stack': 100, 'state': 'participating', 'name': 'p3', 'uuid': 'pdvfqlaedvtdxezzhqzmdb'}
            ]
        }

round_state_call = {'dealer_btn': 0, 'big_blind_pos': 2, 'round_count': 1, 'small_blind_pos': 1, 'next_player': 0, 'small_blind_amount': 5, 'action_histories': {'preflop': [{'action': 'ANTE', 'amount': 10, 'uuid': 'bwivgajwiewdkpymuztbxq'}, {'action': 'ANTE', 'amount': 10, 'uuid': 'pdvfqlaedvtdxezzhqzmdb'}, {'action': 'ANTE', 'amount': 10, 'uuid': 'vzrsrzvmyzaipkcenzdqwp'}, {'action': 'SMALLBLIND', 'amount': 5, 'add_amount': 5, 'uuid': 'bwivgajwiewdkpymuztbxq'}, {'action': 'BIGBLIND', 'amount': 10, 'add_amount': 5, 'uuid': 'pdvfqlaedvtdxezzhqzmdb'}, {'action': 'CALL', 'amount': 10, 'uuid': 'vzrsrzvmyzaipkcenzdqwp', 'paid': 10}]}, 'street': 'preflop', 'seats': [{'stack': 80, 'state': 'participating', 'name': 'p1', 'uuid': 'vzrsrzvmyzaipkcenzdqwp'}, {'stack': 85, 'state': 'participating', 'name': 'p2', 'uuid': 'bwivgajwiewdkpymuztbxq'}, {'stack': 80, 'state': 'participating', 'name': 'p3', 'uuid': 'pdvfqlaedvtdxezzhqzmdb'}], 'community_card': [], 'pot': {'main': {'amount': 55}, 'side': []}}

round_state_raise = {'dealer_btn': 0, 'big_blind_pos': 2, 'round_count': 1, 'small_blind_pos': 1, 'next_player': 1, 'small_blind_amount': 5, 'action_histories': {'preflop': [{'action': 'ANTE', 'amount': 10, 'uuid': 'bwivgajwiewdkpymuztbxq'}, {'action': 'ANTE', 'amount': 10, 'uuid': 'pdvfqlaedvtdxezzhqzmdb'}, {'action': 'ANTE', 'amount': 10, 'uuid': 'vzrsrzvmyzaipkcenzdqwp'}, {'action': 'SMALLBLIND', 'amount': 5, 'add_amount': 5, 'uuid': 'bwivgajwiewdkpymuztbxq'}, {'action': 'BIGBLIND', 'amount': 10, 'add_amount': 5, 'uuid': 'pdvfqlaedvtdxezzhqzmdb'}, {'action': 'CALL', 'amount': 10, 'uuid': 'vzrsrzvmyzaipkcenzdqwp', 'paid': 10}, {'action': 'RAISE', 'amount': 15, 'add_amount': 5, 'paid': 10, 'uuid': 'bwivgajwiewdkpymuztbxq'}]}, 'street': 'preflop', 'seats': [{'stack': 80, 'state': 'participating', 'name': 'p1', 'uuid': 'vzrsrzvmyzaipkcenzdqwp'}, {'stack': 75, 'state': 'participating', 'name': 'p2', 'uuid': 'bwivgajwiewdkpymuztbxq'}, {'stack': 80, 'state': 'participating', 'name': 'p3', 'uuid': 'pdvfqlaedvtdxezzhqzmdb'}], 'community_card': [], 'pot': {'main': {'amount': 65}, 'side': []}}

round_state_allin_raise = {'dealer_btn': 0, 'big_blind_pos': 2, 'round_count': 1, 'small_blind_pos': 1, 'next_player': 2, 'small_blind_amount': 5, 'action_histories': {'preflop': [{'action': 'ANTE', 'amount': 10, 'uuid': 'bwivgajwiewdkpymuztbxq'}, {'action': 'ANTE', 'amount': 10, 'uuid': 'pdvfqlaedvtdxezzhqzmdb'}, {'action': 'ANTE', 'amount': 10, 'uuid': 'vzrsrzvmyzaipkcenzdqwp'}, {'action': 'SMALLBLIND', 'amount': 5, 'add_amount': 5, 'uuid': 'bwivgajwiewdkpymuztbxq'}, {'action': 'BIGBLIND', 'amount': 10, 'add_amount': 5, 'uuid': 'pdvfqlaedvtdxezzhqzmdb'}, {'action': 'CALL', 'amount': 10, 'uuid': 'vzrsrzvmyzaipkcenzdqwp', 'paid': 10}, {'action': 'RAISE', 'amount': 15, 'add_amount': 5, 'paid': 10, 'uuid': 'bwivgajwiewdkpymuztbxq'}, {'action': 'RAISE', 'amount': 90, 'add_amount': 75, 'paid': 80, 'uuid': 'pdvfqlaedvtdxezzhqzmdb'}]}, 'street': 'preflop', 'seats': [{'stack': 80, 'state': 'participating', 'name': 'p1', 'uuid': 'vzrsrzvmyzaipkcenzdqwp'}, {'stack': 75, 'state': 'participating', 'name': 'p2', 'uuid': 'bwivgajwiewdkpymuztbxq'}, {'stack': 0, 'state': 'allin', 'name': 'p3', 'uuid': 'pdvfqlaedvtdxezzhqzmdb'}], 'community_card': [], 'pot': {'main': {'amount': 145}, 'side': [{'amount': 0, 'eligibles': ['pdvfqlaedvtdxezzhqzmdb']}]}}

round_state_allin_call = {'dealer_btn': 0, 'big_blind_pos': 2, 'round_count': 1, 'small_blind_pos': 1, 'next_player': 0, 'small_blind_amount': 5, 'action_histories': {'preflop': [{'action': 'ANTE', 'amount': 10, 'uuid': 'bwivgajwiewdkpymuztbxq'}, {'action': 'ANTE', 'amount': 10, 'uuid': 'pdvfqlaedvtdxezzhqzmdb'}, {'action': 'ANTE', 'amount': 10, 'uuid': 'vzrsrzvmyzaipkcenzdqwp'}, {'action': 'SMALLBLIND', 'amount': 5, 'add_amount': 5, 'uuid': 'bwivgajwiewdkpymuztbxq'}, {'action': 'BIGBLIND', 'amount': 10, 'add_amount': 5, 'uuid': 'pdvfqlaedvtdxezzhqzmdb'}, {'action': 'CALL', 'amount': 10, 'uuid': 'vzrsrzvmyzaipkcenzdqwp', 'paid': 10}, {'action': 'RAISE', 'amount': 15, 'add_amount': 5, 'paid': 10, 'uuid': 'bwivgajwiewdkpymuztbxq'}, {'action': 'RAISE', 'amount': 90, 'add_amount': 75, 'paid': 80, 'uuid': 'pdvfqlaedvtdxezzhqzmdb'}, {'action': 'CALL', 'amount': 90, 'uuid': 'vzrsrzvmyzaipkcenzdqwp', 'paid': 80}]}, 'street': 'preflop', 'seats': [{'stack': 0, 'state': 'allin', 'name': 'p1', 'uuid': 'vzrsrzvmyzaipkcenzdqwp'}, {'stack': 75, 'state': 'participating', 'name': 'p2', 'uuid': 'bwivgajwiewdkpymuztbxq'}, {'stack': 0, 'state': 'allin', 'name': 'p3', 'uuid': 'pdvfqlaedvtdxezzhqzmdb'}], 'community_card': [], 'pot': {'main': {'amount': 225}, 'side': [{'amount': 0, 'eligibles': ['vzrsrzvmyzaipkcenzdqwp', 'pdvfqlaedvtdxezzhqzmdb']}, {'amount': 0, 'eligibles': ['vzrsrzvmyzaipkcenzdqwp', 'pdvfqlaedvtdxezzhqzmdb']}]}}

round_state_fold = {'dealer_btn': 0, 'big_blind_pos': 2, 'round_count': 1, 'small_blind_pos': 1, 'next_player': 1, 'small_blind_amount': 5, 'action_histories': {'preflop': [{'action': 'ANTE', 'amount': 10, 'uuid': 'bwivgajwiewdkpymuztbxq'}, {'action': 'ANTE', 'amount': 10, 'uuid': 'pdvfqlaedvtdxezzhqzmdb'}, {'action': 'ANTE', 'amount': 10, 'uuid': 'vzrsrzvmyzaipkcenzdqwp'}, {'action': 'SMALLBLIND', 'amount': 5, 'add_amount': 5, 'uuid': 'bwivgajwiewdkpymuztbxq'}, {'action': 'BIGBLIND', 'amount': 10, 'add_amount': 5, 'uuid': 'pdvfqlaedvtdxezzhqzmdb'}, {'action': 'CALL', 'amount': 10, 'uuid': 'vzrsrzvmyzaipkcenzdqwp', 'paid': 10}, {'action': 'RAISE', 'amount': 15, 'add_amount': 5, 'paid': 10, 'uuid': 'bwivgajwiewdkpymuztbxq'}, {'action': 'RAISE', 'amount': 90, 'add_amount': 75, 'paid': 80, 'uuid': 'pdvfqlaedvtdxezzhqzmdb'}, {'action': 'CALL', 'amount': 90, 'uuid': 'vzrsrzvmyzaipkcenzdqwp', 'paid': 80}, {'action': 'FOLD', 'uuid': 'bwivgajwiewdkpymuztbxq'}]}, 'street': 'preflop', 'seats': [{'stack': 0, 'state': 'allin', 'name': 'p1', 'uuid': 'vzrsrzvmyzaipkcenzdqwp'}, {'stack': 75, 'state': 'folded', 'name': 'p2', 'uuid': 'bwivgajwiewdkpymuztbxq'}, {'stack': 0, 'state': 'allin', 'name': 'p3', 'uuid': 'pdvfqlaedvtdxezzhqzmdb'}], 'community_card': [], 'pot': {'main': {'amount': 225}, 'side': [{'amount': 0, 'eligibles': ['vzrsrzvmyzaipkcenzdqwp', 'pdvfqlaedvtdxezzhqzmdb']}, {'amount': 0, 'eligibles': ['vzrsrzvmyzaipkcenzdqwp', 'pdvfqlaedvtdxezzhqzmdb']}]}}

