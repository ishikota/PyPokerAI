from pypokerengine.engine.poker_constants import PokerConstants as Const
from pypokerengine.engine.data_encoder import DataEncoder
from pypokerengine.utils.card_utils import gen_cards, estimate_hole_card_win_rate

def construct_scalar_features(round_state, my_uuid, hole_card, blind_strecture):
    f_stack = player_stack_to_scalar
    f_state = player_state_to_scaled_scalar
    f_history = player_action_history_to_scalar
    round_count = round_level_to_scalar(round_state, blind_strecture)
    dealer_btn = dealer_btn_to_scalar(round_state, my_uuid)
    next_player = next_player_to_scalar(round_state, my_uuid)
    sb_pos = sb_pos_to_scalar(round_state, my_uuid)
    street = street_to_scalar(round_state)
    cards = cards_to_scaled_scalar(round_state, hole_card)
    seats = seats_to_vector(round_state, f_stack, f_state, f_history)
    pot = pot_to_scalar(round_state)
    return round_count + dealer_btn + next_player + sb_pos + street + cards + seats + pot

def construct_scaled_scalar_features(round_state, my_uuid, hole_card, blind_strecture):
    f_stack = player_stack_to_scaled_scalar
    f_state = player_state_to_scaled_scalar
    f_history = player_action_history_to_scaled_scalar
    round_count = round_level_to_scaled_scalar(round_state, blind_strecture)
    dealer_btn = dealer_btn_to_scaled_scalar(round_state, my_uuid)
    next_player = next_player_to_scaled_scalar(round_state, my_uuid)
    sb_pos = sb_pos_to_scaled_scalar(round_state, my_uuid)
    street = street_to_scaled_scalar(round_state)
    cards = cards_to_scaled_scalar(round_state, hole_card)
    seats = seats_to_vector(round_state, f_stack, f_state, f_history)
    pot = pot_to_scaled_scalar(round_state)
    return round_count + dealer_btn + next_player + sb_pos + street + cards + seats + pot

def construct_onehot_features(round_state, my_uuid, hole_card, blind_strecture):
    f_stack = player_stack_to_scaled_scalar
    f_state = player_state_to_onehot
    f_history = player_action_history_to_scaled_scalar
    round_count = round_level_to_onehot(round_state, blind_strecture)
    dealer_btn = dealer_btn_to_onehot(round_state, my_uuid)
    next_player = next_player_to_onehot(round_state, my_uuid)
    sb_pos = sb_pos_to_onehot(round_state, my_uuid)
    street = street_to_onehot(round_state)
    cards = cards_to_scaled_scalar(round_state, hole_card)
    seats = seats_to_vector(round_state, f_stack, f_state, f_history)
    pot = pot_to_scaled_scalar(round_state)
    return round_count + dealer_btn + next_player + sb_pos + street + cards + seats + pot


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

def cards_to_scaled_scalar(round_state, hole_card, simulation_num=100):
    player_num = len(round_state["seats"])
    hole = gen_cards(hole_card)
    community = gen_cards(round_state["community_card"])
    return [estimate_hole_card_win_rate(simulation_num, player_num, hole, community)]

def seats_to_vector(round_state, f_stack, f_state, f_history):
    player_num = len(round_state["seats"])
    c_p2vec = lambda pos: player_to_vector(round_state, pos, f_stack, f_state, f_history)
    player_vecs = [c_p2vec(pos) for pos in range(player_num)]
    return reduce(lambda acc, e: acc+e, player_vecs, [])

def player_to_vector(round_state, seat_pos, f_stack, f_state, f_history):
    my_info = round_state["seats"][seat_pos]
    stack = f_stack(round_state, seat_pos)
    state = f_state(round_state, seat_pos)
    history  = f_history(round_state, seat_pos)
    return stack + state + history

def player_stack_to_scalar(round_state, seat_pos):
    my_info = round_state["seats"][seat_pos]
    return [my_info["stack"]]

def player_stack_to_scaled_scalar(round_state, seat_pos):
    player_num = len(round_state["seats"])
    stack_sum = sum([player_stack_to_scalar(round_state, i)[0] for i in range(player_num)])
    pot_amount = pot_to_scalar(round_state)[0]
    all_chip = stack_sum + pot_amount
    return [1.0 * player_stack_to_scalar(round_state, seat_pos)[0] / all_chip]

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
    pot_amount = pot_to_scalar(round_state)[0]
    pay_amount = player_action_history_to_scalar(round_state, seat_pos)[0]
    return [1.0 * pay_amount / pot_amount]

def pot_to_scalar(round_state):
    pot = round_state["pot"]
    return [pot["main"]["amount"] + sum([side["amount"] for side in pot["side"]])]

def pot_to_scaled_scalar(round_state):
    player_num = len(round_state["seats"])
    stack_sum = sum([player_stack_to_scalar(round_state, i)[0] for i in range(player_num)])
    pot_amount = pot_to_scalar(round_state)[0]
    return [1.0 * pot_amount / (pot_amount + stack_sum)]

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

