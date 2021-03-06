import pypokerai.features as F
import pypokerai.task as T

from nose.tools import raises
from mock import patch, Mock
from tests.base_unittest import BaseUnitTest
from tests.sample_data import round_state1, round_state2
from pypokerengine.engine.data_encoder import DataEncoder
from pypokerengine.engine.poker_constants import PokerConstants as Const

class FeaturesTest(BaseUnitTest):

    def test_round_count_to_scalar(self):
        self.eq(3, F.round_count_to_scalar(round_state1)[0])

    def test_round_level_to_scalar(self):
        blind_strecture = {
                11 : { "ante": 0, "small_blind": 50 },
                21 : { "ante": 0, "small_blind": 75 },
                31 : { "ante": 0, "small_blind": 100 }
        }
        self.eq(0, F.round_level_to_scalar(round_state1, blind_strecture)[0])
        self.eq(1, F.round_level_to_scalar({ "round_count": 15 }, blind_strecture)[0])
        self.eq(2, F.round_level_to_scalar({ "round_count": 25 }, blind_strecture)[0])
        self.eq(3, F.round_level_to_scalar({ "round_count": 40 }, blind_strecture)[0])

    def test_round_level_to_scalar(self):
        blind_strecture = {
                11 : { "ante": 0, "small_blind": 50 },
                21 : { "ante": 0, "small_blind": 75 },
                31 : { "ante": 0, "small_blind": 100 }
        }
        self.eq(0, F.round_level_to_scaled_scalar(round_state1, blind_strecture)[0])
        self.almosteq(0.33, F.round_level_to_scaled_scalar({ "round_count": 15 }, blind_strecture)[0], 0.01)
        self.almosteq(0.66, F.round_level_to_scaled_scalar({ "round_count": 25 }, blind_strecture)[0], 0.01)
        self.almosteq(1.00, F.round_level_to_scaled_scalar({ "round_count": 40 }, blind_strecture)[0], 0.01)

    def test_round_level_to_onehot(self):
        blind_strecture = {
                11 : { "ante": 0, "small_blind": 50 },
                21 : { "ante": 0, "small_blind": 75 },
                31 : { "ante": 0, "small_blind": 100 }
        }
        self.eq([1,0,0,0], F.round_level_to_onehot(round_state1, blind_strecture))
        self.eq([0,1,0,0], F.round_level_to_onehot({ "round_count": 15 }, blind_strecture))
        self.eq([0,0,1,0], F.round_level_to_onehot({ "round_count": 25 }, blind_strecture))
        self.eq([0,0,0,1], F.round_level_to_onehot({ "round_count": 40 }, blind_strecture))

    def test_dealer_btn_to_scalar(self):
        self.eq(1, F.dealer_btn_to_scalar(round_state1, "zjwhieqjlowtoogemqrjjo")[0])
        self.eq(2, F.dealer_btn_to_scalar(round_state1, "xgbpujiwtcccyicvfqffgy")[0])
        self.eq(0, F.dealer_btn_to_scalar(round_state1, "pnqfqsvgwkegkuwnzucvxw")[0])

    def test_dealer_btn_to_scaled_scalar(self):
        self.almosteq(0.33, F.dealer_btn_to_scaled_scalar(round_state1, "zjwhieqjlowtoogemqrjjo")[0], 0.01)
        self.almosteq(0.66, F.dealer_btn_to_scaled_scalar(round_state1, "xgbpujiwtcccyicvfqffgy")[0], 0.01)
        self.almosteq(0, F.dealer_btn_to_scaled_scalar(round_state1, "pnqfqsvgwkegkuwnzucvxw")[0], 0.01)

    def test_dealer_btn_to_onehot(self):
        self.eq([0,1,0], F.dealer_btn_to_onehot(round_state1, "zjwhieqjlowtoogemqrjjo"))
        self.eq([0,0,1], F.dealer_btn_to_onehot(round_state1, "xgbpujiwtcccyicvfqffgy"))
        self.eq([1,0,0], F.dealer_btn_to_onehot(round_state1, "pnqfqsvgwkegkuwnzucvxw"))

    def test_next_player_to_scalar(self):
        self.eq(1, F.next_player_to_scalar(round_state1, "zjwhieqjlowtoogemqrjjo")[0])
        self.eq(2, F.next_player_to_scalar(round_state1, "xgbpujiwtcccyicvfqffgy")[0])
        self.eq(0, F.next_player_to_scalar(round_state1, "pnqfqsvgwkegkuwnzucvxw")[0])

    def test_next_player_to_scaled_scalar(self):
        self.almosteq(0.33, F.next_player_to_scaled_scalar(round_state1, "zjwhieqjlowtoogemqrjjo")[0], 0.01)
        self.almosteq(0.66, F.next_player_to_scaled_scalar(round_state1, "xgbpujiwtcccyicvfqffgy")[0], 0.01)
        self.almosteq(0, F.next_player_to_scaled_scalar(round_state1, "pnqfqsvgwkegkuwnzucvxw")[0], 0.01)

    def test_next_player_to_onehot(self):
        self.eq([0,1,0], F.next_player_to_onehot(round_state1, "zjwhieqjlowtoogemqrjjo"))
        self.eq([0,0,1], F.next_player_to_onehot(round_state1, "xgbpujiwtcccyicvfqffgy"))
        self.eq([1,0,0], F.next_player_to_onehot(round_state1, "pnqfqsvgwkegkuwnzucvxw"))

    def test_sb_pos_to_scalar(self):
        self.eq(0, F.sb_pos_to_scalar(round_state1, "zjwhieqjlowtoogemqrjjo")[0])
        self.eq(1, F.sb_pos_to_scalar(round_state1, "xgbpujiwtcccyicvfqffgy")[0])
        self.eq(2, F.sb_pos_to_scalar(round_state1, "pnqfqsvgwkegkuwnzucvxw")[0])

    def test_sb_pos_to_scaled_scalar(self):
        self.almosteq(0, F.sb_pos_to_scaled_scalar(round_state1, "zjwhieqjlowtoogemqrjjo")[0], 0.01)
        self.almosteq(0.33, F.sb_pos_to_scaled_scalar(round_state1, "xgbpujiwtcccyicvfqffgy")[0], 0.01)
        self.almosteq(0.66, F.sb_pos_to_scaled_scalar(round_state1, "pnqfqsvgwkegkuwnzucvxw")[0], 0.01)

    def test_sb_pos_to_onehot(self):
        self.eq([1,0,0], F.sb_pos_to_onehot(round_state1, "zjwhieqjlowtoogemqrjjo"))
        self.eq([0,1,0], F.sb_pos_to_onehot(round_state1, "xgbpujiwtcccyicvfqffgy"))
        self.eq([0,0,1], F.sb_pos_to_onehot(round_state1, "pnqfqsvgwkegkuwnzucvxw"))

    @raises(ValueError)
    def test_street_to_onehot_raise(self):
        F.street_to_scalar({ "street": "showdown" })

    def test_street_to_scalar(self):
        self.eq(0, F.street_to_scalar({ "street": "preflop" })[0])
        self.eq(1, F.street_to_scalar({ "street": "flop" })[0])
        self.eq(2, F.street_to_scalar({ "street": "turn" })[0])
        self.eq(3, F.street_to_scalar({ "street": "river" })[0])

    def test_street_to_scaled_scalar(self):
        self.almosteq(0, F.street_to_scaled_scalar({ "street": "preflop" })[0], 0.01)
        self.almosteq(0.33, F.street_to_scaled_scalar({ "street": "flop" })[0], 0.01)
        self.almosteq(0.66, F.street_to_scaled_scalar({ "street": "turn" })[0], 0.01)
        self.almosteq(1.0, F.street_to_scaled_scalar({ "street": "river" })[0], 0.01)

    def test_street_to_onehot(self):
        self.eq([1,0,0,0], F.street_to_onehot({ "street": "preflop" }))
        self.eq([0,1,0,0], F.street_to_onehot({ "street": "flop" }))
        self.eq([0,0,1,0], F.street_to_onehot({ "street": "turn" }))
        self.eq([0,0,0,1], F.street_to_onehot({ "street": "river" }))

    # FIXME patch does not work so this test fails
    def xtest_cards_to_scaled_scalar(self):
        with patch("pypokerengine.utils.card_utils.estimate_hole_card_win_rate", side_effect=[0.1]) as f:
            self.eq(0.1, F.cards_to_scaled_scalar(round_state1, ["S2", "D4"])[0])

    def xtest_cards_to_scaled_scalar_by_neuralnet(self):
        from holecardhandicapper.model.neuralnet import Neuralnet
        nns = [Neuralnet("preflop"), Neuralnet("flop"), Neuralnet("turn"), Neuralnet("river")]
        [nn.compile() for nn in nns]
        self.almosteq(0.56, F.cards_to_scaled_scalar(round_state1, ["S2", "D4"], "neuralnet", neuralnets=nns)[0], 0.01)
        self.almosteq(0.20, F.cards_to_scaled_scalar(round_state2, ["S2", "D4"], "neuralnet", neuralnets=nns)[0], 0.01)

    def xtest_cards_to_binary_array(self):
        from holecardhandicapper.model.neuralnet import Neuralnet
        nns = [Neuralnet("preflop"), Neuralnet("flop"), Neuralnet("turn"), Neuralnet("river")]
        [nn.compile() for nn in nns]
        expected = [0, 0, 1, 0, 1, 1, 0, 0, 0, 1]  # bin(564) => 0b1000110100
        self.eq(expected, F.cards_to_binary_array(round_state1, ["S2", "D4"], "neuralnet", neuralnets=nns))

    def test_player_stack_to_scalar(self):
        self.eq(80, F.player_stack_to_scalar(round_state1, 0)[0])
        self.eq(0, F.player_stack_to_scalar(round_state1, 1)[0])
        self.eq(120, F.player_stack_to_scalar(round_state1, 2)[0])

    def test_player_stack_to_scaled_scalar(self):
        self.almosteq(0.26, F.player_stack_to_scaled_scalar(round_state1, 0)[0], 0.01)
        self.almosteq(0, F.player_stack_to_scaled_scalar(round_state1, 1)[0], 0.01)
        self.almosteq(0.40, F.player_stack_to_scaled_scalar(round_state1, 2)[0], 0.01)

    def test_player_stack_to_binary_array(self):
        expected1 = [1, 1, 0, 1, 0, 0, 0, 0, 1, 0]  # bin(267) => 100000100
        expected2 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        expected3 = [0, 0, 0, 0, 1, 0, 0, 1, 1, 0]
        self.eq(expected1, F.player_stack_to_binary_array(round_state1, 0))
        self.eq(expected2, F.player_stack_to_binary_array(round_state1, 1))
        self.eq(expected3, F.player_stack_to_binary_array(round_state1, 2))

    def test_player_state_to_scaled_scalar(self):
        self.eq(1, F.player_state_to_scaled_scalar(round_state1, 0)[0])
        self.eq(1, F.player_state_to_scaled_scalar(round_state1, 1)[0])
        self.eq(1, F.player_state_to_scaled_scalar(round_state1, 2)[0])

    def test_player_state_to_onehot(self):
        self.eq([0,1], F.player_state_to_onehot(round_state1, 0))
        self.eq([0,1], F.player_state_to_onehot(round_state1, 1))
        self.eq([0,1], F.player_state_to_onehot(round_state1, 2))

    def test_player_action_history_to_scalar(self):
        self.eq(35, F.player_action_history_to_scalar(round_state1, 0)[0])
        self.eq(35, F.player_action_history_to_scalar(round_state1, 1)[0])
        self.eq(35, F.player_action_history_to_scalar(round_state1, 2)[0])

    def test_player_action_history_to_scaled_scalar(self):
        self.almosteq(0.116, F.player_action_history_to_scaled_scalar(round_state1, 0)[0], 0.001)
        self.almosteq(0.116, F.player_action_history_to_scaled_scalar(round_state1, 1)[0], 0.001)
        self.almosteq(0.116, F.player_action_history_to_scaled_scalar(round_state1, 2)[0], 0.001)

    def test_player_action_history_to_binary_array(self):
        expected = [1, 0, 1, 0, 1, 1, 1, 0, 0, 0]
        self.eq(expected, F.player_action_history_to_binary_array(round_state1, 0))
        self.eq(expected, F.player_action_history_to_binary_array(round_state1, 1))
        self.eq(expected, F.player_action_history_to_binary_array(round_state1, 2))

    def test_player_action_record_to_ratio(self):
        record1 = [[],[50],[75],[]]
        expected1 = [0, 0.5, 0.5, 0]
        record2 = [[0,0],[],[],[10000,100]]
        expected2 = [0.5, 0, 0, 0.5]
        self.eq(expected1, F.player_action_record_to_action_ratio(record1, 0))
        self.eq(expected2, F.player_action_record_to_action_ratio(record2, 0))

    def test_player_action_record_to_ratio(self):
        record1 = [[],[50],[75],[]]
        expected1 = [0, 0.1, 0.1, 0]
        record2 = [[0,0],[],[],[10000,100]]
        expected2 = [0.2, 0, 0, 0.2]
        self.eq(expected1, F.player_action_record_to_action_ratio(record1, 10))
        self.eq(expected2, F.player_action_record_to_action_ratio(record2, 10))

    def test_player_to_vector(self):
        f_stack = F.player_stack_to_scalar
        f_state = F.player_state_to_onehot
        f_history = F.player_action_history_to_scaled_scalar
        vec = F.player_to_vector(round_state1, 0, f_stack, f_state, f_history)
        self.size(4, vec)
        self.almosteq([80, 0, 1, 0.116], vec, 0.001)

    def test_player_to_vector_with_action_record(self):
        f_stack = F.player_stack_to_scaled_scalar
        f_state = F.player_state_to_scaled_scalar
        f_history = F.player_action_history_to_scaled_scalar
        record = {"zjwhieqjlowtoogemqrjjo": [[0], [50], [100], [10000, 10000]]}
        vec = F.player_to_vector(round_state1, 0, f_stack, f_state, f_history, record)
        self.size(7, vec)
        self.almosteq([0.26, 1, 0.116, 0.2, 0.2, 0.2, 0.4], vec, 0.01)

    def test_seats_to_vector(self):
        f_stack = F.player_stack_to_scaled_scalar
        f_state = F.player_state_to_scaled_scalar
        f_history = F.player_action_history_to_scaled_scalar
        vec1 = F.seats_to_vector(round_state1, f_stack, f_state, f_history, "zjwhieqjlowtoogemqrjjo")
        vec2 = F.seats_to_vector(round_state1, f_stack, f_state, f_history, "xgbpujiwtcccyicvfqffgy")
        vec3 = F.seats_to_vector(round_state1, f_stack, f_state, f_history, "pnqfqsvgwkegkuwnzucvxw")
        self.size(9, vec1)
        self.almosteq([0.26, 1, 0.116, 0, 1, 0.116, 0.4, 1, 0.116], vec1, 0.01)
        self.size(9, vec2)
        self.almosteq([0, 1, 0.116, 0.4, 1, 0.116, 0.26, 1, 0.116], vec2, 0.01)
        self.size(9, vec3)
        self.almosteq([0.4, 1, 0.116, 0.26, 1, 0.116, 0, 1, 0.116], vec3, 0.01)

    def test_seats_to_vector_with_action_record(self):
        f_stack = F.player_stack_to_scaled_scalar
        f_state = F.player_state_to_scaled_scalar
        f_history = F.player_action_history_to_scaled_scalar
        action_record = {
            "zjwhieqjlowtoogemqrjjo": [[],[50],[100],[]],
            "xgbpujiwtcccyicvfqffgy": [[],[],[100],[]],
            "pnqfqsvgwkegkuwnzucvxw": [[0],[],[],[]]
            }
        vec1 = F.seats_to_vector(round_state1, f_stack, f_state, f_history, "zjwhieqjlowtoogemqrjjo", action_record)
        self.size(21, vec1)
        self.almosteq([
            0.26, 1, 0.116, 0, 0.5, 0.5, 0,
            0, 1, 0.116, 0, 0, 1.0, 0,
            0.4, 1, 0.116, 1.0, 0, 0, 0
            ], vec1, 0.01)

    def test_pot_to_scalar(self):
        self.eq(100, F.pot_to_scalar(round_state1)[0])

    def test_pot_to_scaled_scalar(self):
        self.almosteq(0.33, F.pot_to_scaled_scalar(round_state1)[0], 0.01)

    def test_pot_to_binary_array(self):
        expected = [1, 0, 1, 1, 0, 0, 1, 0, 1, 0]
        self.eq(expected, F.pot_to_binary_array(round_state1))

    def test_action_to_onehot(self):
        self.eq([1,0,0,0,0,0], F.action_to_onehot(T.gen_fold_action()))
        self.eq([0,1,0,0,0,0], F.action_to_onehot(T.gen_call_action(10)))
        self.eq([0,0,1,0,0,0], F.action_to_onehot(T.gen_min_raise_action(10)))
        self.eq([0,0,0,1,0,0], F.action_to_onehot(T.gen_double_raise_action(10)))
        self.eq([0,0,0,0,1,0], F.action_to_onehot(T.gen_triple_raise_action(10)))
        self.eq([0,0,0,0,0,1], F.action_to_onehot(T.gen_max_raise_action(10)))

    def test_binary_array_negative_value(self):
        self.eq([0]*10, F._small_number_to_binary_array(-1))

    def test_construct_scalar_features(self):
        action = T.gen_fold_action()
        blind_structure = { 1: "dummy", 3: "dummy", 5: "dummy", 10: "dummy" }
        vec = F.construct_scalar_features(round_state1, "zjwhieqjlowtoogemqrjjo", ["S2", "D4"], blind_structure, action, algorithm="simulation")
        self.size(14, vec)

    def test_construct_scaled_scalar_features(self):
        action = T.gen_fold_action()
        blind_structure = { 1: "dummy", 3: "dummy", 5: "dummy", 10: "dummy" }
        vec = F.construct_scaled_scalar_features(round_state1, "zjwhieqjlowtoogemqrjjo", ["S2", "D4"], blind_structure, action, algorithm="simulation")
        self.size(14, vec)

    def test_construct_scaled_scalar_features_with_action_record(self):
        task = T.TexasHoldemTask(final_round=10, action_record=True)
        def recommend_random_action(state, action):
            return 1
        value_func = Mock()
        value_func.predict_value.side_effect = recommend_random_action
        task.set_opponent_value_functions([value_func]*9)
        state = task.generate_initial_state()
        round_state = DataEncoder.encode_round_state(state)
        blind_structure = { 1: "dummy", 3: "dummy", 5: "dummy", 10: "dummy" }
        act_call = task.generate_possible_actions(state)[1]
        state = task.transit_state(state, act_call)
        #state = task.transit_state(state, act_call)
        #self.stop()
        vec = F.construct_scaled_scalar_features_with_action_record(
                state, round_state, T.my_uuid, ["S2", "D4"], blind_structure, "dummy_action", algorithm="simulation")
        self.size(35+10*4, vec)

    def test_construct_onehot_features(self):
        action = T.gen_fold_action()
        blind_structure = { 1: "dummy", 3: "dummy", 5: "dummy", 10: "dummy" }
        vec = F.construct_onehot_features(round_state1, "zjwhieqjlowtoogemqrjjo", ["S2", "D4"], blind_structure, action, algorithm="simulation")
        self.size(26, vec)

    def test_construct_binary_onehot_features(self):
        action = T.gen_fold_action()
        blind_structure = { 1: "dummy", 3: "dummy", 5: "dummy", 10: "dummy" }
        vec = F.construct_binary_onehot_features(round_state1, "zjwhieqjlowtoogemqrjjo", ["S2", "D4"], blind_structure, action, algorithm="simulation")
        self.size(98, vec)

    def xtest_visualize_scalar_features_weight(self):
        import numpy as np
        title = F.scalar_features_title()
        acts = ["FOLD", "CALL", "MIN_RAISE", "DOUBLE_RAISE", "TRIPLE_RAISE", "MAX_RAISE"]
        weights = [["%s_%s" % (act, t) for t in title] for act in acts]
        np_w = np.array(weights).T
        visualized = F.visualize_scalar_features_weight([np_w], debug=True)
        self.stop()

    def xtest_visualize_scaled_scalar_features_weight(self):
        import numpy as np
        title = F.scalar_features_title()
        acts = ["FOLD", "CALL", "MIN_RAISE", "DOUBLE_RAISE", "TRIPLE_RAISE", "MAX_RAISE"]
        weights = [["%s_%s" % (act, t) for t in title] for act in acts]
        np_w = np.array(weights).T
        visualized = F.visualize_scaled_scalar_features_weight([np_w], debug=True)
        self.stop()

    def xtest_visualize_onehot_features_weight(self):
        import numpy as np
        title = F.onehot_features_title()
        acts = ["FOLD", "CALL", "MIN_RAISE", "DOUBLE_RAISE", "TRIPLE_RAISE", "MAX_RAISE"]
        weights = [["%s_%s" % (act, t) for t in title] for act in acts]
        np_w = np.array(weights).T
        visualized = F.visualize_onehot_features_weight([np_w], debug=True)
        self.stop()

    def xtest_visualize_binary_onehot_features_weight(self):
        import numpy as np
        title = F.binary_onehot_features_title()
        acts = ["FOLD", "CALL", "MIN_RAISE", "DOUBLE_RAISE", "TRIPLE_RAISE", "MAX_RAISE"]
        weights = [["%s_%s" % (act, t) for t in title] for act in acts]
        np_w = np.array(weights).T
        visualized = F.visualize_binary_onehot_features_weight([np_w], debug=True)
        self.stop()

