from pypokerai.player import PokerPlayer, ConsolePlayer
from pypokerai.task import TexasHoldemTask, blind_structure
from pypokerengine.api.game import setup_config, start_poker

def play_game(max_round, names, value_functions, with_me=False, verbose=1):
    if with_me:
        assert len(value_functions) == 9
    else:
        assert len(value_functions) == 10
    task = TexasHoldemTask()
    config = setup_config(max_round=max_round, initial_stack=10000, small_blind_amount=25)
    config.set_blind_structure(blind_structure)
    for name, value_func in zip(names, value_functions):
        config.register_player(name, PokerPlayer(task, value_func, debug=verbose==1))
    if with_me:
        config.register_player("console", ConsolePlayer())
    return start_poker(config, verbose=verbose)

