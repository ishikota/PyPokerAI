import os

from kyoka.utils import pickle_data, unpickle_data, value_function_check, build_not_implemented_msg
from kyoka.value_function import BaseTabularActionValueFunction, BaseApproxActionValueFunction
from kyoka.algorithm.rl_algorithm import BaseRLAlgorithm, generate_episode
from kyoka.algorithm.deep_q_learning import ExperienceReplay,\
        choose_action, predict_value, initialize_replay_memory


class DeepSarsa(BaseRLAlgorithm):
    """Basic "on-policy" Temporal-Difference Learning method.

    "on-policy" indicates that this method uses same policy when
    "select action during the episode" and "create backup target".
    (backup target is the target value used when training value function)

    Algorithm is implemented based on the book "Reinforcement Learning: An Introduction"
    (reference : https://webdocs.cs.ualberta.ca/~sutton/book/bookdraft2016sep.pdf)

    - Algorithm -
    Initialize
        T  <- your RL task
        PI <- policy used in the algorithm
        Q  <- action value function
        a  <- learning rate (alpha)
        g  <- discounting factor (gamma)
    Repeat until computational budge runs out:
        S <- generate initial state of task T
        A <- choose action at S by following policy PI
        Repeat until S is terminal state:
            S' <- next state of S after taking action A
            R <- reward gained by taking action A at state S
            A' <- next action at S' by following policy PI
            Q(S, A) <- Q(S, A) + a * [ R + g * Q(S', A') - Q(S, A)]
            S, A <- S', A'
    """

    SAVE_FILE_NAME = "deep_sarsa_replay_memory.pickle"

    def __init__(
            self, alpha=0.1, gamma=0.9, N=1000000, minibatch_size=32, replay_start_size=50000):
        """
        Args:
            alpha: learning rate. default=0.1. 0 < alpha <= 1
            gamma: discounting factor. default=0.9. 0 < gamma <= 1
            N <- capacity of replay memory
            minibatch_size <- size of minibatch used to train Q
            replay_start_size <- initial size of replay memory.
        """
        self.alpha = alpha
        self.gamma = gamma
        self.replay_memory = ExperienceReplay(max_size=N)
        self.minibatch_size = minibatch_size
        self.replay_start_size = replay_start_size

    def setup(self, task, policy, value_function):
        validate_value_function(value_function)
        super(DeepSarsa, self).setup(task, policy, value_function)
        initialize_replay_memory(task, value_function, self.replay_memory, self.replay_start_size)

    def run_gpi_for_an_episode(self, task, policy, value_function):
        state = task.generate_initial_state()
        action = policy.choose_action(task, value_function, state)
        while not task.is_terminal_state(state):
            next_state = task.transit_state(state, action)
            next_action = choose_action(task, policy, value_function, next_state)
            reward = task.calculate_reward(next_state)
            self.replay_memory.store_transition(state, action, reward, next_state)

            experience_minibatch = self.replay_memory.sample_minibatch(self.minibatch_size)
            backup_minibatch = self._gen_backup_minibatch(
                    task, policy, value_function, experience_minibatch)
            value_function.backup_on_minibatch(backup_minibatch)
            state, action = next_state, next_action

    def save_algorithm_state(self, save_dir_path):
        """Save initial params, replay memory"""
        state = (
                self.gamma, self.replay_memory.dump(), self.minibatch_size, self.replay_start_size
                )
        pickle_data(self._gen_replay_memory_save_path(save_dir_path), state)

    def load_algorithm_state(self, load_dir_path):
        """Load initial params, replay memory"""
        state = unpickle_data(self._gen_replay_memory_save_path(load_dir_path))
        (self.gamma, replay_memory_serial, self.minibatch_size, self.replay_start_size) = state
        new_replay_memory = ExperienceReplay(max_size=1)
        new_replay_memory.load(replay_memory_serial)
        self.replay_memory = new_replay_memory

    def _gen_backup_minibatch(self, task, policy, value_function, experience_minibatch):
        """Create minibatch of backup targets from minibatch of experiences
        Returns
            backup_minibatch : minibatch of training data for value function.
                               It's array of learning data which is tuple of
                               (state, action, backup_target).
                               Most of the case value function is trained by
                               using MSE between Q(state, action) and backup_target.
        """
        backup_minibatch = [
                self._gen_backup_data(task, policy, value_function, experience)
                for experience in experience_minibatch]
        return backup_minibatch

    def _gen_backup_data(self, task, policy, value_function, experience):
        """Transform experience into backup targets in training data format
        Returns
            learning data: tuple of (state, action, backup_target).
                           value function receives minibatch of this tuples and
                           train value function maybe like below.
                           MSE <- ( Q(state, action) - backup_target )^2
        """
        state, action, reward, next_state = experience
        next_action = choose_action(task, policy, value_function, next_state)
        next_Q_value = predict_value(value_function, next_state, next_action)
        backup_target = reward + self.gamma * next_Q_value
        return (state, action, backup_target)

    def _gen_replay_memory_save_path(self, dir_path):
        return os.path.join(dir_path, self.SAVE_FILE_NAME)

class DeepSarsaApproxActionValueFunction(BaseApproxActionValueFunction):

    def backup_on_minibatch(self, backup_minibatch):
        """Define how to train Q network which you defined
        Args:
            backup_minibatch : minibatch of training data for Q. It's array of
                               learning data which is tuple of
                               (state, action, backup_target).
                               Most of the case value function is trained by
                               using MSE between Q(state, action) and backup_target.
        """
        err_msg = build_not_implemented_msg(self, "backup_on_minibatch")
        raise NotImplementedError(err_msg)

def validate_value_function(value_function):
    value_function_check("Sarsa",
            [DeepSarsaApproxActionValueFunction],
            value_function)

