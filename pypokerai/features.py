from pypokerengine.engine.poker_constants import PokerConstants as Const
from pypokerengine.engine.data_encoder import DataEncoder
from pypokerengine.utils.card_utils import gen_cards, estimate_hole_card_win_rate

from pypokerai.task import FOLD, CALL, MIN_RAISE, DOUBLE_RAISE, TRIPLE_RAISE, MAX_RAISE, ACTION_RECORD_KEY

def construct_scalar_features(round_state, my_uuid, hole_card, blind_strecture, neuralnets=None, algorithm="neuralnet"):
    f_stack = player_stack_to_scalar
    f_state = player_state_to_scaled_scalar
    f_history = player_action_history_to_scalar
    round_count = round_level_to_scalar(round_state, blind_strecture)
    dealer_btn = dealer_btn_to_scalar(round_state, my_uuid)
    street = street_to_scalar(round_state)
    cards = cards_to_scaled_scalar(round_state, hole_card, algorithm, neuralnets=neuralnets)
    seats = seats_to_vector(round_state, f_stack, f_state, f_history, my_uuid)
    pot = pot_to_scalar(round_state)
    return round_count + dealer_btn + street + cards + seats + pot

def visualize_scalar_features_weight(weights, debug=False):
    assert weights[0].shape == (35, 6)
    acts = ["FOLD", "CALL", "MIN_RAISE", "DOUBLE_RAISE", "TRIPLE_RAISE", "MAX_RAISE"]
    w_for_acts = weights[0].T
    if not debug: w_for_acts = [[round(e,4) for e in weight] for weight in w_for_acts]
    ls = []
    ls.append("round count")
    for act, w in zip(acts, w_for_acts): ls.append("  %s : %s" % (act, w[0]))
    ls.append("dealer button")
    for act, w in zip(acts, w_for_acts): ls.append("  %s : %s" % (act, w[1]))
    ls.append("street")
    for act, w in zip(acts, w_for_acts): ls.append("  %s : %s" % (act, w[2]))
    ls.append("cards")
    for act, w in zip(acts, w_for_acts): ls.append("  %s : %s" % (act, w[3]))
    ls.append("pot")
    for act, w in zip(acts, w_for_acts): ls.append("  %s : %s" % (act, w[34]))
    ls.append("seats")
    for i in range(10):
        stack, state, history = {}, {}, {}
        for act, w in zip(acts, w_for_acts):
            w_player = w[4+i*3:4+i*3+3]
            stack[act] = w_player[0]
            state[act] = w_player[1]
            history[act] = w_player[2]
        ls.append("    player %d" % i)
        ls.append("      %s : %s" % ("stack", stack))
        ls.append("      %s : %s" % ("state", state))
        ls.append("      %s : %s" % ("history(paid sum)", history))
    return "\n".join(ls)

def scalar_features_title():
    ts = []
    ts.append("round level")
    ts.append("btn distance")
    ts.append("street")
    ts.append("cards")
    for i in range(10):
        player = "player-%d" % i
        for title in ["stack", "state", "history"]:
            ts.append("%s:%s" % (player, title))
    ts.append("pot")
    return ts

def construct_scaled_scalar_features(round_state, my_uuid, hole_card, blind_strecture, neuralnets=None, algorithm="neuralnet"):
    f_stack = player_stack_to_scaled_scalar
    f_state = player_state_to_scaled_scalar
    f_history = player_action_history_to_scaled_scalar
    round_count = round_level_to_scaled_scalar(round_state, blind_strecture)
    dealer_btn = dealer_btn_to_scaled_scalar(round_state, my_uuid)
    street = street_to_scaled_scalar(round_state)
    cards = cards_to_scaled_scalar(round_state, hole_card, algorithm, neuralnets=neuralnets)
    seats = seats_to_vector(round_state, f_stack, f_state, f_history, my_uuid)
    pot = pot_to_scaled_scalar(round_state)
    return round_count + dealer_btn + street + cards + seats + pot

def visualize_scaled_scalar_features_weight(weights, debug=False):
    return visualize_scalar_features_weight(weights, debug)

def scaled_scalar_features_title():
    return scalar_features_title()

def construct_scaled_scalar_features_with_action_record(
        state, round_state, my_uuid, hole_card, blind_strecture, neuralnets=None, algorithm="neuralnet"):
    assert state.has_key(ACTION_RECORD_KEY)
    f_stack = player_stack_to_scaled_scalar
    f_state = player_state_to_scaled_scalar
    f_history = player_action_history_to_scaled_scalar
    round_count = round_level_to_scaled_scalar(round_state, blind_strecture)
    dealer_btn = dealer_btn_to_scaled_scalar(round_state, my_uuid)
    street = street_to_scaled_scalar(round_state)
    cards = cards_to_scaled_scalar(round_state, hole_card, algorithm, neuralnets=neuralnets)
    seats = seats_to_vector(round_state, f_stack, f_state, f_history, my_uuid, action_record=state[ACTION_RECORD_KEY], action_record_logic_threthold=10)
    pot = pot_to_scaled_scalar(round_state)
    return round_count + dealer_btn + street + cards + seats + pot

def construct_onehot_features(round_state, my_uuid, hole_card, blind_strecture, neuralnets=None, algorithm="neuralnet"):
    f_stack = player_stack_to_scaled_scalar
    f_state = player_state_to_onehot
    f_history = player_action_history_to_scaled_scalar
    round_count = round_level_to_onehot(round_state, blind_strecture)
    dealer_btn = dealer_btn_to_onehot(round_state, my_uuid)
    street = street_to_onehot(round_state)
    cards = cards_to_scaled_scalar(round_state, hole_card, algorithm, neuralnets=neuralnets)
    seats = seats_to_vector(round_state, f_stack, f_state, f_history, my_uuid)
    pot = pot_to_scaled_scalar(round_state)
    return round_count + dealer_btn + street + cards + seats + pot

def visualize_onehot_features_weight(weights, debug=False):
    assert weights[0].shape == (83, 6)
    acts = ["FOLD", "CALL", "MIN_RAISE", "DOUBLE_RAISE", "TRIPLE_RAISE", "MAX_RAISE"]
    w_for_acts = weights[0].T
    if not debug: w_for_acts = [[round(e,4) for e in weight] for weight in w_for_acts]
    ls = []
    ls.append("round count")
    for act, w in zip(acts, w_for_acts): ls.append("  %s : %s" % (act, w[:27]))
    ls.append("dealer button")
    for act, w in zip(acts, w_for_acts): ls.append("  %s : %s" % (act, w[27:37]))
    ls.append("street")
    for act, w in zip(acts, w_for_acts): ls.append("  %s : %s" % (act, w[37:41]))
    ls.append("cards")
    for act, w in zip(acts, w_for_acts): ls.append("  %s : %s" % (act, w[41]))
    ls.append("pot")
    for act, w in zip(acts, w_for_acts): ls.append("  %s : %s" % (act, w[82]))
    ls.append("seats")
    for i in range(10):
        stack, state, history = {}, {}, {}
        for act, w in zip(acts, w_for_acts):
            w_player = w[42+i*4:42+i*4+4]
            stack[act] = w_player[0]
            state[act] = w_player[1:3]
            history[act] = w_player[3]
        ls.append("    player %d" % i)
        ls.append("      %s : %s" % ("stack", stack))
        ls.append("      %s : %s" % ("state", state))
        ls.append("      %s : %s" % ("history(paid sum)", history))
    return "\n".join(ls)

def onehot_features_title():
    ts = []
    for i in range(27):
        ts.append("round level %d" % (i+1))
    for i in range(10):
        ts.append("btn distance=%d" % i)
    for street in ["preflop","flop","turn","river"]:
        ts.append("street-%s" % street)
    ts.append("cards")
    for i in range(10):
        player = "player-%d" % i
        for title in ["stack", "state-fold", "state-active", "history"]:
            ts.append("%s:%s" % (player, title))
    ts.append("pot")
    return ts

def construct_binary_onehot_features(
        round_state, my_uuid, hole_card, blind_strecture, neuralnets=None, algorithm="neuralnet"):
    f_stack = player_stack_to_binary_array
    f_state = player_state_to_onehot
    f_history = player_action_history_to_binary_array
    round_count = round_level_to_onehot(round_state, blind_strecture)
    dealer_btn = dealer_btn_to_onehot(round_state, my_uuid)
    street = street_to_onehot(round_state)
    cards = cards_to_binary_array(round_state, hole_card, algorithm, neuralnets=neuralnets)
    seats = seats_to_vector(round_state, f_stack, f_state, f_history, my_uuid)
    pot = pot_to_binary_array(round_state)
    return round_count + dealer_btn + street + cards + seats + pot

def visualize_binary_onehot_features_weight(weights, debug=False):
    assert weights[0].shape == (281, 6)
    acts = ["FOLD", "CALL", "MIN_RAISE", "DOUBLE_RAISE", "TRIPLE_RAISE", "MAX_RAISE"]
    w_for_acts = weights[0].T
    if debug:
        w_for_acts = w_for_acts.tolist()
    else:
         w_for_acts = [[round(e,4) for e in weight] for weight in w_for_acts]
    ls = []
    ls.append("round count")
    for act, w in zip(acts, w_for_acts): ls.append("  %s : %s" % (act, w[:27]))
    ls.append("dealer button")
    for act, w in zip(acts, w_for_acts): ls.append("  %s : %s" % (act, w[27:37]))
    ls.append("street")
    for act, w in zip(acts, w_for_acts): ls.append("  %s : %s" % (act, w[37:41]))
    ls.append("cards")
    for act, w in zip(acts, w_for_acts): ls.append("  %s : %s" % (act, w[41:51]))
    ls.append("pot")
    for act, w in zip(acts, w_for_acts): ls.append("  %s : %s" % (act, w[271:281]))
    ls.append("seats")
    for i in range(10):
        stack, state, history = {}, {}, {}
        for act, w in zip(acts, w_for_acts):
            w_player = w[51+i*22:51+i*22+22]
            stack[act] = w_player[:10]
            state[act] = w_player[10:12]
            history[act] = w_player[12:]
        ls.append("    player %d" % i)
        ls.append("      stack")
        for act in acts:
            ls.append("        %s : %s" % (act, stack[act]))
        ls.append("      state")
        for act in acts:
            ls.append("        %s : %s" % (act, state[act]))
        ls.append("      history")
        for act in acts:
            ls.append("        %s : %s" % (act, history[act]))
    return "\n".join(ls)

def binary_onehot_features_title():
    BINARY_LENGTH = 10
    ts = []
    for i in range(27):
        ts.append("round level %d" % (i+1))
    for i in range(10):
        ts.append("btn distance=%d" % i)
    for street in ["preflop","flop","turn","river"]:
        ts.append("street-%s" % street)
    for i in range(BINARY_LENGTH):
        ts.append("cards-%d bit" % i)
    for i in range(10):
        player = "player-%d" % i
        for i in range(BINARY_LENGTH):
            ts.append("%s:stack-%d bit" % (player, i))
        for state in ["fold", "active"]:
            ts.append("%s:%s" % (player, state))
        for i in range(BINARY_LENGTH):
            ts.append("%s:history-%d bit" % (player, i))
    for i in range(10):
        ts.append("pot-%d bit" % i)
    return ts

def round_count_to_scalar(round_state):
    round_count = round_state["round_count"]
    return [round_count]

def round_level_to_scalar(round_state, blind_strecture):
    round_count = round_state["round_count"]
    level_threshold = sorted(blind_strecture.keys())
    current_level_pos = [round_count >= threshold for threshold in level_threshold].count(True)
    return [current_level_pos]

def round_level_to_scaled_scalar(round_state, blind_strecture):
    current_level_pos = round_level_to_scalar(round_state, blind_strecture)[0]
    level_num = len(blind_strecture.keys())
    assert level_num != 0
    return [1.0 * current_level_pos / level_num]

def round_level_to_onehot(round_state, blind_strecture):
    size = len(blind_strecture.keys()) + 1
    current_level_pos = round_level_to_scalar(round_state, blind_strecture)[0]
    return [1 if current_level_pos == idx else 0 for idx in range(size)]

def dealer_btn_to_scalar(round_state, my_uuid):
    return [_measure_distance_to_me(round_state, my_uuid, round_state["dealer_btn"])]

def dealer_btn_to_scaled_scalar(round_state, my_uuid):
    distance_to_btn = dealer_btn_to_scalar(round_state, my_uuid)[0]
    return [_distance_to_scaled_scalar(round_state, distance_to_btn)]

def dealer_btn_to_onehot(round_state, my_uuid):
    distance_to_btn = dealer_btn_to_scalar(round_state, my_uuid)[0]
    return _distance_to_onehot(round_state, distance_to_btn)

def next_player_to_scalar(round_state, my_uuid):
    return [_measure_distance_to_me(round_state, my_uuid, round_state["next_player"])]

def next_player_to_scaled_scalar(round_state, my_uuid):
    distance_to_next_player = next_player_to_scalar(round_state, my_uuid)[0]
    return [_distance_to_scaled_scalar(round_state, distance_to_next_player)]

def next_player_to_onehot(round_state, my_uuid):
    distance_to_next_player = next_player_to_scalar(round_state, my_uuid)[0]
    return _distance_to_onehot(round_state, distance_to_next_player)

def sb_pos_to_scalar(round_state, my_uuid):
    return [_measure_distance_to_me(round_state, my_uuid, round_state["small_blind_pos"])]

def sb_pos_to_scaled_scalar(round_state, my_uuid):
    distance_to_sb_pos = sb_pos_to_scalar(round_state, my_uuid)[0]
    return [_distance_to_scaled_scalar(round_state, distance_to_sb_pos)]

def sb_pos_to_onehot(round_state, my_uuid):
    distance_to_sb_pos = sb_pos_to_scalar(round_state, my_uuid)[0]
    return _distance_to_onehot(round_state, distance_to_sb_pos)

def street_to_scalar(round_state):
    if "preflop" == round_state["street"]: return [0]
    if "flop" == round_state["street"]: return [1]
    if "turn" == round_state["street"]: return [2]
    if "river" == round_state["street"]: return [3]
    raise ValueError("Invalid street [ %s ] received" % round_state["street"])

def street_to_scaled_scalar(round_state):
    street = street_to_scalar(round_state)[0]
    return [1.0 * street / 3]

def street_to_onehot(round_state):
    street = street_to_scalar(round_state)[0]
    return [1 if street == idx else 0 for idx in range(4)]

def cards_to_scaled_scalar(round_state, hole_card, algorithm, simulation_num=100, neuralnets=None):
    if "simulation" == algorithm:
        return cards_to_scaled_scalar_by_simulation(round_state, hole_card, simulation_num)
    if "neuralnet" == algorithm:
        return cards_to_scaled_scalar_by_neuralnet(round_state, hole_card, neuralnets)
    else:
        raise ValueError("Unexpected flg [ algorithm=%s ] received" % algorithm)

def cards_to_scaled_scalar_by_simulation(round_state, hole_card, simulation_num):
    player_num = len(round_state["seats"])
    hole = gen_cards(hole_card)
    community = gen_cards(round_state["community_card"])
    return [estimate_hole_card_win_rate(simulation_num, player_num, hole, community)]

def cards_to_scaled_scalar_by_neuralnet(round_state, hole_card, neuralnets):
    hole = gen_cards(hole_card)
    community = gen_cards(round_state["community_card"])
    if "preflop" == round_state["street"]:
        return [neuralnets[0].predict(hole)]
    if "flop" == round_state["street"]:
        return [neuralnets[1].predict(hole, community)]
    if "turn" == round_state["street"]:
        return [neuralnets[2].predict(hole, community)]
    if "river" == round_state["street"]:
        return [neuralnets[3].predict(hole, community)]
    raise Exception("Unexpected street [ %s ] received" % round_state["street"])

def cards_to_binary_array(round_state, hole_card, algorithm, simulation_num=100, neuralnets=None):
    return _small_number_to_binary_array(
            cards_to_scaled_scalar(round_state, hole_card, algorithm, simulation_num, neuralnets)[0]
            )

def seats_to_vector(round_state, f_stack, f_state, f_history, my_uuid, action_record=None,
        action_record_logic_threthold=0):
    my_pos = [p["uuid"] for p in round_state["seats"]].index(my_uuid)
    player_num = len(round_state["seats"])
    relative_pos = range(player_num) + range(player_num)
    relative_pos = relative_pos[my_pos:my_pos+player_num]
    c_p2vec = lambda pos: player_to_vector(
            round_state, pos, f_stack, f_state, f_history, action_record, action_record_logic_threthold)
    player_vecs = [c_p2vec(pos) for pos in relative_pos]
    return reduce(lambda acc, e: acc+e, player_vecs, [])

def player_to_vector(round_state, seat_pos, f_stack, f_state, f_history, action_record=None,
        action_record_logic_threthold=0):
    my_info = round_state["seats"][seat_pos]
    stack = f_stack(round_state, seat_pos)
    state = f_state(round_state, seat_pos)
    history  = f_history(round_state, seat_pos)
    vec = stack + state + history
    if action_record:
        my_act_record = action_record[my_info["uuid"]]
        record_vec = player_action_record_to_action_ratio(my_act_record, action_record_logic_threthold)
        vec += record_vec
    return vec

def player_action_record_to_action_ratio(player_act_record, logic_threthold):
    act_counts = [len(amounts) for amounts in player_act_record]
    all_counts = sum(act_counts)
    if all_counts >= logic_threthold:
        return [1.0*count/all_counts if all_counts!=0 else 0 for count in act_counts]
    else:
        return [0.1*count for count in act_counts]

def player_stack_to_scalar(round_state, seat_pos):
    my_info = round_state["seats"][seat_pos]
    return [my_info["stack"]]

def player_stack_to_scaled_scalar(round_state, seat_pos):
    player_num = len(round_state["seats"])
    stack_sum = sum([player_stack_to_scalar(round_state, i)[0] for i in range(player_num)])
    pot_amount = pot_to_scalar(round_state)[0]
    all_chip = stack_sum + pot_amount
    return [1.0 * player_stack_to_scalar(round_state, seat_pos)[0] / all_chip]

def player_stack_to_binary_array(round_state, seat_pos):
    return _small_number_to_binary_array(
            player_stack_to_scaled_scalar(round_state, seat_pos)[0])

def player_state_to_scaled_scalar(round_state, seat_pos):
    state = round_state["seats"][seat_pos]["state"]
    return [0] if state == DataEncoder.PAY_INFO_FOLDED_STR else [1]

def player_state_to_onehot(round_state, seat_pos):
    state_scalar = player_state_to_scaled_scalar(round_state, seat_pos)[0]
    return [1 if state_scalar==idx else 0 for idx in range(2)]

def player_action_history_to_scalar(round_state, seat_pos):
    uuid = round_state["seats"][seat_pos]["uuid"]
    round_histories = round_state["action_histories"]
    player_histories = { street: [h for h in round_histories[street] if h["uuid"]==uuid]\
            for street in round_histories.keys() }
    player_histories = { k:v for k,v in player_histories.items() if len(v)!=0 }
    player_last_actions = { street: player_histories[street][-1] for street in player_histories.keys() }
    pay_sum = sum([history["amount"] for history in player_last_actions.values() if history["action"]!="FOLD"])
    if player_histories.has_key("preflop"):
        ante_history = [h for h in player_histories["preflop"] if h["action"]=="ANTE"]
    else:
        ante_history = []
    if len(ante_history) != 0:
        pay_sum += ante_history[0]["amount"]
    return [pay_sum]

def player_action_history_to_scaled_scalar(round_state, seat_pos):
    player_num = len(round_state["seats"])
    stack_sum = sum([player_stack_to_scalar(round_state, i)[0] for i in range(player_num)])
    pot_amount = pot_to_scalar(round_state)[0]
    all_chip = stack_sum + pot_amount
    pay_amount = player_action_history_to_scalar(round_state, seat_pos)[0]
    return [1.0 * pay_amount / all_chip]

def player_action_history_to_binary_array(round_state, seat_pos):
    return _small_number_to_binary_array(
            player_action_history_to_scaled_scalar(round_state, seat_pos)[0])

def pot_to_scalar(round_state):
    pot = round_state["pot"]
    return [pot["main"]["amount"] + sum([side["amount"] for side in pot["side"]])]

def pot_to_scaled_scalar(round_state):
    player_num = len(round_state["seats"])
    stack_sum = sum([player_stack_to_scalar(round_state, i)[0] for i in range(player_num)])
    pot_amount = pot_to_scalar(round_state)[0]
    return [1.0 * pot_amount / (pot_amount + stack_sum)]

def pot_to_binary_array(round_state):
    return _small_number_to_binary_array(pot_to_scaled_scalar(round_state)[0])

def action_to_onehot(action):
    actions = [FOLD, CALL, MIN_RAISE, DOUBLE_RAISE, TRIPLE_RAISE, MAX_RAISE]
    idx = actions.index(action["name"])
    return [1 if idx==i else 0 for i in range(len(actions))]

def _small_number_to_binary_array(num):
    num = max(0, min(0.999, num))
    significant = int(round(num*1000))
    return [int(b) for b in bin(significant)[2:].zfill(10)][::-1]

def _measure_distance_to_me(round_state, my_uuid, start_pos):
    player_num = len(round_state["seats"])
    my_pos = [p["uuid"] for p in round_state["seats"]].index(my_uuid)
    if my_pos < start_pos: my_pos += player_num
    distance_to_me = my_pos - start_pos
    return distance_to_me

def _distance_to_scaled_scalar(round_state, distance):
    player_num = len(round_state["seats"])
    return 1.0 * distance / player_num

def _distance_to_onehot(round_state, distance):
    player_num = len(round_state["seats"])
    return [1 if distance == idx else 0 for idx in range(player_num)]

