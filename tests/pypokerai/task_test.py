from mock import Mock
from tests.base_unittest import BaseUnitTest
from pypokerai.task import TexasHoldemTask, my_uuid, pick_me, blind_structure
from pypokerengine.engine.action_checker import ActionChecker
from pypokerengine.engine.data_encoder import DataEncoder
from pypokerengine.engine.poker_constants import PokerConstants as Const

class OneRoundPokerTaskTest(BaseUnitTest):

    def setUp(self):
        self.task = TexasHoldemTask(final_round=1)
        def recommend_fold(state, action):
            if action["action"] == "fold":
                return 1
            else:
                return 0
        value_func = Mock()
        value_func.predict_value.side_effect = recommend_fold
        self.task.set_opponent_value_functions([value_func]*9)

    def test_generate_initial_state(self):
        state = self.task.generate_initial_state()
        me = pick_me(state)
        players = state["table"].seats.players
        self.eq(1, state["round_count"])
        self.eq(25, state["small_blind_amount"])
        self.eq(0, state["street"])
        self.eq(players.index(me), state["next_player"])
        self.size(10, players)
        self.eq(10000, players[0].stack)
        self.eq(9975, players[1].stack)
        self.eq(9950, players[2].stack)
        self.true(all([p.stack == 10000 for p in state["table"].seats.players[3:]]))
        self.include("agent", [p.name for p in state["table"].seats.players])

    def test_is_terminal_state(self):
        state = self.task.generate_initial_state()
        self.false(self.task.is_terminal_state(state))
        state["street"] = Const.Street.FINISHED
        self.true(self.task.is_terminal_state(state))

    def test_transit_state_when_others_folded(self):
        state = self.task.generate_initial_state()
        act_call = self.task.generate_possible_actions(state)[1]
        state = self.task.transit_state(state, act_call)
        self.eq(10075, pick_me(state).stack)
        self.true(self.task.is_terminal_state(state))
        self.eq(10075, self.task.calculate_reward(state))

    def test_transit_state_when_agent_folded(self):
        state = self.task.generate_initial_state()
        act_fold = self.task.generate_possible_actions(state)[0]
        state = self.task.transit_state(state, act_fold)
        self.eq(10000, pick_me(state).stack)
        self.true(self.task.is_terminal_state(state))

    def test_transit_state_till_round_finish(self):
        self.task = TexasHoldemTask(final_round=1)
        def recommend_call(state, action):
            if action["action"] == "call":
                return 1
            else:
                return 0
        value_func = Mock()
        value_func.predict_value.side_effect = recommend_call
        self.task.set_opponent_value_functions([value_func]*9)

        state = self.task.generate_initial_state()
        act_call = self.task.generate_possible_actions(state)[1]
        state = self.task.transit_state(state, act_call)
        players = state["table"].seats.players
        round_state = DataEncoder.encode_round_state(state)
        self.eq(1, state["round_count"])
        self.eq(0, state["table"].dealer_btn)
        self.eq(6, state["next_player"])
        self.eq("flop", round_state["street"])
        self.eq(500, round_state["pot"]["main"]["amount"])

        act_call = self.task.generate_possible_actions(state)[1]
        state = self.task.transit_state(state, act_call)
        round_state = DataEncoder.encode_round_state(state)
        self.eq(6, state["next_player"])
        self.eq("turn", round_state["street"])
        self.eq(500, round_state["pot"]["main"]["amount"])

        act_call = self.task.generate_possible_actions(state)[1]
        state = self.task.transit_state(state, act_call)
        round_state = DataEncoder.encode_round_state(state)
        self.eq(6, state["next_player"])
        self.eq("river", round_state["street"])
        self.eq(500, round_state["pot"]["main"]["amount"])

        act_call = self.task.generate_possible_actions(state)[1]
        state = self.task.transit_state(state, act_call)
        self.true(self.task.is_terminal_state(state))

class TexasHoldemTaskTest(BaseUnitTest):

    def setUp(self):
        self.task = TexasHoldemTask()
        def recommend_fold(state, action):
            if action["action"] == "fold":
                return 1
            else:
                return 0
        value_func = Mock()
        value_func.predict_value.side_effect = recommend_fold
        self.task.set_opponent_value_functions([value_func]*9)

    def test_shuffle_seat_position(self):
        def gen_task():
            task = TexasHoldemTask(shuffle_position=True)
            def recommend_fold(state, action):
                if action["action"] == "fold":
                    return 1
                else:
                    return 0
            value_func = Mock()
            value_func.predict_value.side_effect = recommend_fold
            task.set_opponent_value_functions([value_func]*9)
            return task
        # Fail this test in probability 0.01
        test = [6==gen_task().generate_initial_state()["next_player"] for _ in range(10)]
        self.include(False, test)

    def test_generate_initial_state(self):
        state = self.task.generate_initial_state()
        me = pick_me(state)
        players = state["table"].seats.players
        self.eq(1, state["round_count"])
        self.eq(25, state["small_blind_amount"])
        self.eq(0, state["street"])
        self.eq(players.index(me), state["next_player"])
        self.size(10, players)
        self.eq(10000, players[0].stack)
        self.eq(9975, players[1].stack)
        self.eq(9950, players[2].stack)
        self.true(all([p.stack == 10000 for p in state["table"].seats.players[3:]]))
        self.include("agent", [p.name for p in state["table"].seats.players])

    def test_generate_possible_actions(self):
        state = self.task.generate_initial_state()
        actions = self.task.generate_possible_actions(state)
        self.size(6, actions)
        self.eq({ "name": "fold", "action": "fold", "amount": 0 }, actions[0])
        self.eq({ "name": "call", "action": "call", "amount": 50 }, actions[1])
        self.eq({ "name": "min_raise", "action": "raise", "amount": 75 }, actions[2])
        self.eq({ "name": "double_raise", "action": "raise", "amount": 150 }, actions[3])
        self.eq({ "name": "triple_raise", "action": "raise", "amount": 225 }, actions[4])
        self.eq({ "name": "max_raise", "action": "raise", "amount": 10000 }, actions[5])

        # Check if generated actions are valid
        correct = lambda act, amount: ActionChecker.correct_action(state["table"].seats.players, 2, 25, act, amount)
        for action in [a for a in actions if not a["action"]=="fold"]:
            self.neq("fold", correct(action["action"], action["amount"])[0])

    def test_transit_state(self):
        state = self.task.generate_initial_state()
        act_fold = self.task.generate_possible_actions(state)[0]
        state = self.task.transit_state(state, act_fold)
        players = state["table"].seats.players
        self.eq(2, state["round_count"])
        self.eq(1, state["table"].dealer_btn)
        self.eq(6, state["next_player"])
        self.eq(9975, players[1].stack)
        self.eq(10000, players[2].stack)
        self.eq(9950, players[3].stack)

    def test_transit_state_when_agent_lose(self):
        # if agent cards is stronger than other 9 players this test fails
        def recommend_call(state, action):
            if action["action"] == "call":
                return 1
            else:
                return 0
        value_func = Mock()
        value_func.predict_value.side_effect = recommend_call
        self.task.set_opponent_value_functions([value_func]*9)

        state = self.task.generate_initial_state()
        actions = self.task.generate_possible_actions(state)
        allin = [act for act in actions if act["name"] == "max_raise"][0]
        state = self.task.transit_state(state, allin)
        self.true(self.task.is_terminal_state(state))
        self.eq(0, self.task.calculate_reward(state))

    def test_transit_state_when_last_trhee_player(self):
        def recommend_allin(state, action):
            if action["name"] == "max_raise":
                return 1
            if action["amount"] == 10000 and action["name"] == "call":
                return 1
            else:
                return 0
        value_func = Mock()
        value_func.predict_value.side_effect = recommend_allin
        self.task.set_opponent_value_functions([value_func]*9)

        state = self.task.generate_initial_state()
        actions = self.task.generate_possible_actions(state)
        fold = actions[0]
        state = self.task.transit_state(state, fold)
        players = state["table"].seats.players
        # When more than 3 player's card has same strength, below assertion fails
        self.assertLessEqual(len([p for p in players if p.stack !=0]), 3)
        self.true(self.task.is_terminal_state(state))
        self.eq(10000, self.task.calculate_reward(state))

    def test_transit_state_when_round_finish(self):
        state = self.task.generate_initial_state()
        self.eq(1, state["round_count"])
        fold = self.task.generate_possible_actions(state)[0]
        state = self.task.transit_state(state, fold)
        self.eq(2, state["round_count"])
        fold = self.task.generate_possible_actions(state)[0]
        state = self.task.transit_state(state, fold)
        self.eq(3, state["round_count"])

    def test_transit_state_when_final_round(self):
        self.task = TexasHoldemTask(final_round=2)
        def recommend_fold(state, action):
            if action["action"] == "fold":
                return 1
            else:
                return 0
        value_func = Mock()
        value_func.predict_value.side_effect = recommend_fold
        self.task.set_opponent_value_functions([value_func]*9)

        state = self.task.generate_initial_state()
        self.eq(1, state["round_count"])
        fold = self.task.generate_possible_actions(state)[0]
        state = self.task.transit_state(state, fold)
        self.eq(2, state["round_count"])
        fold = self.task.generate_possible_actions(state)[0]
        state = self.task.transit_state(state, fold)
        self.eq(2, state["round_count"])
        self.eq(Const.Street.FINISHED, state["street"])

    def test_is_terminal_state_when_active_player_is_three(self):
        state = self.task.generate_initial_state()
        self.false(self.task.is_terminal_state(state))
        others = [p for p in state["table"].seats.players if p.uuid != my_uuid]
        for player in others[3:]: player.stack = 0
        self.false(self.task.is_terminal_state(state))
        state["street"] = Const.Street.FINISHED
        self.false(self.task.is_terminal_state(state))
        others[0].stack = 0
        self.true(self.task.is_terminal_state(state))

    def test_is_terminal_state_when_me_is_loser(self):
        state = self.task.generate_initial_state()
        me = [p for p in state["table"].seats.players if p.uuid == my_uuid][0]
        state["street"] = Const.Street.FINISHED
        self.false(self.task.is_terminal_state(state))
        me.stack = 0
        self.true(self.task.is_terminal_state(state))

    def test_calculate_reward_when_not_terminal(self):
        state = self.task.generate_initial_state()
        self.eq(0, self.task.calculate_reward(state))

    def test_calculate_reward_when_terminal(self):
        state = self.task.generate_initial_state()
        state["street"] = Const.Street.FINISHED
        others = [p for p in state["table"].seats.players if p.uuid != my_uuid]
        for player in others: player.stack = 0
        self.eq(10000, self.task.calculate_reward(state))

    def test_calculate_reward_scaled_model(self):
        state = self.task.generate_initial_state()
        state["street"] = Const.Street.FINISHED
        others = [p for p in state["table"].seats.players if p.uuid != my_uuid]
        for player in others: player.stack = 0
        task = TexasHoldemTask(scale_reward=True)
        self.eq(0.1, task.calculate_reward(state))

    def test_calculate_reward_lose_penalty(self):
        state = self.task.generate_initial_state()
        me = [p for p in state["table"].seats.players if p.uuid == my_uuid][0]
        state["street"] = Const.Street.FINISHED
        self.false(self.task.is_terminal_state(state))
        me.stack = 0
        self.true(self.task.is_terminal_state(state))
        task = TexasHoldemTask(lose_penalty=True)
        self.eq(-1, task.calculate_reward(state))
        task = TexasHoldemTask(scale_reward=True, lose_penalty=True)
        self.eq(-1, task.calculate_reward(state))

    def test_blind_structure(self):
        bs = blind_structure
        def check(level, ante, sb):
            self.eq(ante, bs[level]["ante"])
            self.eq(sb, bs[level]["small_blind"])
        check(11, 0, 50)
        check(21, 0, 75)
        check(31, 0, 100)
        check(41, 25, 100)
        check(51, 25, 150)
        check(61, 50, 200)
        check(71, 50, 250)
        check(81, 75, 300)
        check(91, 100, 400)
        check(101, 100, 600)
        check(111, 200, 800)
        check(121, 200, 1000)
        check(131, 300, 1200)
        check(141, 400, 1500)
        check(151, 500, 2000)
        check(161, 500, 2500)
        check(171, 500, 3000)
        check(181, 1000, 4000)
        check(191, 1000, 6000)
        check(201, 2000, 8000)
        check(211, 3000, 10000)
        check(221, 4000, 12000)
        check(231, 5000, 15000)
        check(241, 10000, 20000)
        check(251, 20000, 30000)

    def test_generate_initial_state_with_play_history(self):
        task = TexasHoldemTask(final_round=10, action_record=True)
        def recommend_call(state, action):
            if action["name"] == "call":
                return 1
            else:
                return 0
        value_func = Mock()
        value_func.predict_value.side_effect = recommend_call
        task.set_opponent_value_functions([value_func]*9)
        state = task.generate_initial_state()
        h = state["players_action_record"]
        self.eq(10, len(h))
        uuids = [p.uuid for p in state["table"].seats.players]
        self.eq([], [r["name"] for r in h[uuids[0]]])
        self.eq([], [r["name"] for r in h[uuids[1]]])
        self.eq([], [r["name"] for r in h[uuids[2]]])
        self.eq(["call"], [r["name"] for r in h[uuids[3]]])
        self.eq(["call"], [r["name"] for r in h[uuids[4]]])
        self.eq(["call"], [r["name"] for r in h[uuids[5]]])
        self.eq([], [r["name"] for r in h[uuids[6]]])
        self.eq([], [r["name"] for r in h[uuids[7]]])
        self.eq([], [r["name"] for r in h[uuids[8]]])
        self.eq([], [r["name"] for r in h[uuids[9]]])

    def test_transit_state_with_play_history(self):
        task = TexasHoldemTask(final_round=10, action_record=True)
        def recommend_fold(state, action):
            if action["name"] == "fold":
                return 1
            else:
                return 0
        value_func = Mock()
        value_func.predict_value.side_effect = recommend_fold
        task.set_opponent_value_functions([value_func]*9)
        state = task.generate_initial_state()
        act_call = self.task.generate_possible_actions(state)[1]
        state = task.transit_state(state, act_call)
        h = state["players_action_record"]
        self.eq(10, len(h))
        uuids = [p.uuid for p in state["table"].seats.players]
        self.eq(["fold"], [r["name"] for r in h[uuids[0]]])
        self.eq(["fold"], [r["name"] for r in h[uuids[1]]])
        self.eq(["fold"], [r["name"] for r in h[uuids[2]]])
        self.eq(["fold"], [r["name"] for r in h[uuids[3]]])
        self.eq(["fold", "fold"], [r["name"] for r in h[uuids[4]]])
        self.eq(["fold", "fold"], [r["name"] for r in h[uuids[5]]])
        self.eq(["call"], [r["name"] for r in h[uuids[6]]])
        self.eq(["fold"], [r["name"] for r in h[uuids[7]]])
        self.eq(["fold"], [r["name"] for r in h[uuids[8]]])
        self.eq(["fold"], [r["name"] for r in h[uuids[9]]])

        act_raise = self.task.generate_possible_actions(state)[2]
        state = task.transit_state(state, act_raise)
        h = state["players_action_record"]
        self.eq(10, len(h))
        self.eq(["fold", "fold"], [r["name"] for r in h[uuids[0]]])
        self.eq(["fold", "fold"], [r["name"] for r in h[uuids[1]]])
        self.eq(["fold", "fold"], [r["name"] for r in h[uuids[2]]])
        self.eq(["fold", "fold"], [r["name"] for r in h[uuids[3]]])
        self.eq(["fold", "fold"], [r["name"] for r in h[uuids[4]]])
        self.eq(["fold", "fold", "fold"], [r["name"] for r in h[uuids[5]]])
        self.eq(["call", "min_raise"], [r["name"] for r in h[uuids[6]]])
        self.eq(["fold", "fold"], [r["name"] for r in h[uuids[7]]])
        self.eq(["fold", "fold"], [r["name"] for r in h[uuids[8]]])
        self.eq(["fold", "fold"], [r["name"] for r in h[uuids[9]]])

