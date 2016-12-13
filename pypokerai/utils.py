from pypokerai.player import PokerPlayer, ConsolePlayer
from pypokerai.task import TexasHoldemTask, blind_structure
from pypokerengine.api.game import setup_config, start_poker

def play_game(value_functions, with_me=False):
    if with_me:
        assert len(value_functions) == 9
    else:
        assert len(value_functions) == 10
    task = TexasHoldemTask()
    config = setup_config(max_round=10000, initial_stack=10000, small_blind_amount=25)
    config.set_blind_structure(blind_structure)
    for idx, value_func in enumerate(value_functions):
        config.register_player("cpu-%d" % idx, PokerPlayer(task, value_func))
    if with_me:
        config.register_player("console", ConsolePlayer())
    return start_poker(config, verbose=1)

