import pypokerai.features as F
import pypokerai.task as T

from nose.tools import raises
from mock import patch
from tests.base_unittest import BaseUnitTest
from tests.sample_data import round_state1, round_state2
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

    def test_player_stack_to_scalar(self):
        self.eq(80, F.player_stack_to_scalar(round_state1, 0)[0])
        self.eq(0, F.player_stack_to_scalar(round_state1, 1)[0])
        self.eq(120, F.player_stack_to_scalar(round_state1, 2)[0])

    def test_player_stack_to_scaled_scalar(self):
        self.almosteq(0.26, F.player_stack_to_scaled_scalar(round_state1, 0)[0], 0.01)
        self.almosteq(0, F.player_stack_to_scaled_scalar(round_state1, 1)[0], 0.01)
        self.almosteq(0.40, F.player_stack_to_scaled_scalar(round_state1, 2)[0], 0.01)

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
        self.eq(0.35, F.player_action_history_to_scaled_scalar(round_state1, 0)[0])
        self.eq(0.35, F.player_action_history_to_scaled_scalar(round_state1, 1)[0])
        self.eq(0.35, F.player_action_history_to_scaled_scalar(round_state1, 2)[0])

    def test_player_to_vector(self):
        f_stack = F.player_stack_to_scalar
        f_state = F.player_state_to_onehot
        f_history = F.player_action_history_to_scaled_scalar
        vec = F.player_to_vector(round_state1, 0, f_stack, f_state, f_history)
        self.size(4, vec)
        self.eq([80, 0, 1, 0.35], vec)

    def test_seats_to_vector(self):
        f_stack = F.player_stack_to_scaled_scalar
        f_state = F.player_state_to_scaled_scalar
        f_history = F.player_action_history_to_scaled_scalar
        vec1 = F.seats_to_vector(round_state1, f_stack, f_state, f_history, "zjwhieqjlowtoogemqrjjo")
        vec2 = F.seats_to_vector(round_state1, f_stack, f_state, f_history, "xgbpujiwtcccyicvfqffgy")
        vec3 = F.seats_to_vector(round_state1, f_stack, f_state, f_history, "pnqfqsvgwkegkuwnzucvxw")
        self.size(9, vec1)
        self.almosteq([0.26, 1, 0.35, 0, 1, 0.35, 0.4, 1, 0.35], vec1, 0.01)
        self.size(9, vec2)
        self.almosteq([0, 1, 0.35, 0.4, 1, 0.35, 0.26, 1, 0.35], vec2, 0.01)
        self.size(9, vec3)
        self.almosteq([0.4, 1, 0.35, 0.26, 1, 0.35, 0, 1, 0.35], vec3, 0.01)

    def test_pot_to_scalar(self):
        self.eq(100, F.pot_to_scalar(round_state1)[0])

    def test_pot_to_scaled_scalar(self):
        self.almosteq(0.33, F.pot_to_scaled_scalar(round_state1)[0], 0.01)

    def test_action_to_onehot(self):
        self.eq([1,0,0,0,0,0], F.action_to_onehot(T.gen_fold_action()))
        self.eq([0,1,0,0,0,0], F.action_to_onehot(T.gen_call_action(10)))
        self.eq([0,0,1,0,0,0], F.action_to_onehot(T.gen_min_raise_action(10)))
        self.eq([0,0,0,1,0,0], F.action_to_onehot(T.gen_double_raise_action(10)))
        self.eq([0,0,0,0,1,0], F.action_to_onehot(T.gen_triple_raise_action(10)))
        self.eq([0,0,0,0,0,1], F.action_to_onehot(T.gen_max_raise_action(10)))

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

    def test_construct_onehot_features(self):
        action = T.gen_fold_action()
        blind_structure = { 1: "dummy", 3: "dummy", 5: "dummy", 10: "dummy" }
        vec = F.construct_onehot_features(round_state1, "zjwhieqjlowtoogemqrjjo", ["S2", "D4"], blind_structure, action, algorithm="simulation")
        self.size(26, vec)

