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
        vec = F.seats_to_vector(round_state1, f_stack, f_state, f_history)
        self.size(9, vec)
        self.almosteq([0.26, 1, 0.35, 0, 1, 0.35, 0.4, 1, 0.35], vec, 0.01)

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
        self.size(22, vec)

    def xtest_visualize_scalar_features_weight(self):
        str_f = F.visualize_scalar_features_weight(scalar_feature_weight)
        self.stop()

    def test_construct_scaled_scalar_features(self):
        action = T.gen_fold_action()
        blind_structure = { 1: "dummy", 3: "dummy", 5: "dummy", 10: "dummy" }
        vec = F.construct_scaled_scalar_features(round_state1, "zjwhieqjlowtoogemqrjjo", ["S2", "D4"], blind_structure, action, algorithm="simulation")
        self.size(22, vec)

    def xtest_visualize_scaled_scalar_features_weight(self):
        str_f = F.visualize_scaled_scalar_features_weight(scaled_feature_weight)
        self.stop()

    def test_construct_onehot_features(self):
        action = T.gen_fold_action()
        blind_structure = { 1: "dummy", 3: "dummy", 5: "dummy", 10: "dummy" }
        vec = F.construct_onehot_features(round_state1, "zjwhieqjlowtoogemqrjjo", ["S2", "D4"], blind_structure, action, algorithm="simulation")
        self.size(38, vec)

    def xtest_visualize_scaled_scalar_features_weight(self):
        str_f = F.visualize_onehot_features_weight(onehot_feature_weight)
        self.stop()

scalar_feature_weight = [
        0.19952522218227386, 0.11049935966730118, 0.29065170884132385, -0.24148307740688324, 0.12393298745155334,
        -0.07699505984783173, -0.18234306573867798, 0.12346580624580383, 0.24412810802459717, 0.12436593323945999,
        0.057397909462451935, -0.19284209609031677, 0.2717251181602478, 0.23752349615097046, 0.051685821264982224,
        -0.19545677304267883, -0.17991232872009277, 0.14493770897388458, -0.2827906310558319, 0.34340640902519226,
        -0.20055130124092102, -0.05734150856733322, 0.2945440411567688, -0.1706457883119583, 0.15003004670143127,
        0.2555035948753357, 0.3465326726436615, -0.2737915813922882, 0.2668386399745941, -0.3227251470088959,
        -0.03943124786019325, 0.2996900677680969, 0.3123629093170166, 0.301430344581604, -0.2186562865972519,
        0.06907488405704498, -0.1493629664182663, -0.196060448884964, 0.022386584430933, 0.25603601336479187,
        0.016076309606432915, 0.1664496660232544, -0.2851199507713318]

scaled_feature_weight = [
        -0.1341409981250763, 0.39049193263053894, 0.13888783752918243, -0.2085663229227066, -0.3137187659740448,
        0.1713169813156128, 0.17714284360408783, -0.31638675928115845, 0.3749939203262329, -0.2319038361310959,
        0.3525644540786743, -0.039587486535310745, -0.10077806562185287, 0.18056946992874146, 0.17574305832386017,
        -0.07251419126987457, -0.26710304617881775, 0.2891980707645416, 0.18143615126609802, 0.039204154163599014,
        -0.022120894864201546, -0.07287611812353134, 0.2106190174818039, 0.21840618550777435, -0.24976478517055511,
        0.09379170835018158, 0.4049249589443207, -0.10645714402198792, -0.00515022873878479, 0.3078053891658783,
        -0.1250043660402298, -0.08672955632209778, 0.37714141607284546, -0.04192765802145004, 0.09068359434604645,
        0.23283618688583374, -0.3021746277809143, 0.14594584703445435, 0.13008718192577362, 0.08283521980047226,
        -0.22441844642162323, -0.1526040881872177, 0.051125023514032364]

onehot_feature_weight = [
        -0.17075195908546448, 0.04178226739168167, 0.07204403728246689, 0.1144314557313919, 0.1599515825510025,
        0.2303985208272934, 0.2334885448217392, 0.2418479323387146, 0.16565914452075958, 0.10675747692584991,
        0.05636655539274216, 0.05502850189805031, -0.01905371993780136, -0.03564739227294922, -0.10790110379457474,
        -0.010343654081225395, 0.10979403555393219, -0.2238711714744568, 0.11265537142753601, -0.09077193588018417,
        0.12765328586101532, -0.0821346789598465, 0.03160339966416359, 0.2080119401216507, -0.16587074100971222,
        0.20481406152248383, 0.010338488966226578, -0.018709028139710426, 0.09642874449491501, 0.032619547098875046,
        0.008147110231220722, 0.013918810524046421, 0.03160124272108078, 0.03911731764674187, 0.03136671707034111,
        0.03649022802710533, 0.054650548845529556, -0.06408976763486862, 0.004910181742161512, 0.013957236893475056,
        -0.14601922035217285, -0.15216800570487976, 0.0021822787821292877, 0.19369421899318695, -0.01618952490389347,
        -0.016470393165946007, 0.2038390189409256, -0.0673506110906601, 0.0027021944988518953, 0.019698314368724823,
        -0.015808021649718285, -0.019478507339954376, -0.03358754515647888, -0.02852635644376278, -0.022207222878932953,
        -0.006241343449801207, 0.055964816361665726, 0.09114661067724228, 0.09631667286157608, 0.05686560645699501,
        0.06775183975696564, 0.11894405633211136, 0.07870654761791229, -0.11944729089736938, -0.15262754261493683,
        0.1314108520746231, -0.1103716641664505, -0.014362169429659843, 0.0068659535609185696, 0.07291378825902939,
        -0.054754167795181274, 0.17020991444587708, 0.17054612934589386, 0.02721034362912178, 0.15275858342647552,
        0.15652266144752502, 0.14440099895000458, 0.017020106315612793, -0.12607327103614807, -0.019979381933808327,
        -0.0027639747131615877, 0.05047088861465454, -0.1412506103515625, 0.06476008892059326, 0.025746069848537445,
        0.10410169512033463, 0.7743445038795471, 0.22451089322566986, -0.11440664529800415, 0.1348922997713089,
        -0.23941852152347565, 0.12759335339069366, 0.13463306427001953, 0.044219374656677246, -0.1451997607946396,
        0.08478983491659164, 0.08001630008220673, 0.10670391470193863, -0.04349938780069351, -0.015293337404727936,
        -0.01657683029770851, 0.0577780045568943, -0.04892614483833313, 0.06116485595703125, 0.038118574768304825,
        0.08863098919391632, 0.10776633024215698, 0.07414474338293076, 0.06671133637428284]

