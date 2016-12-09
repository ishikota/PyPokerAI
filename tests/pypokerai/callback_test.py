import os
from mock import Mock

from tests.base_unittest import BaseUnitTest

from pypokerai.task import TexasHoldemTask
from pypokerai.callback import ResetOpponentValueFunction

class ResetOpponentValueFunctionTest(BaseUnitTest):

    def setUp(self):
        self.task = TexasHoldemTask()

    def test_after_update(self):
        generator = lambda : Mock()
        callback = ResetOpponentValueFunction(checkpoint_dir_path(), 10, generator)
        callback.after_update(9, self.task, "dummy")
        for value_function in self.task.opponent_value_functions.values():
            self.assertIsNone(value_function)
        callback.after_update(10, self.task, "dummy")
        expected = os.path.join(checkpoint_dir_path(), "after_130_iteration")
        for value_function in self.task.opponent_value_functions.values():
            value_function.load.assert_called_with(expected)


def checkpoint_dir_path():
    return os.path.join(os.path.dirname(__file__), "..", "test_checkpoint")

