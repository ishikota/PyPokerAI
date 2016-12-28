import os
import pypokerai.features as F
import numpy as np
from keras.models import Sequential, model_from_json
from keras.layers.core import Dense
from kyoka.value_function import BaseApproxActionValueFunction
from holecardhandicapper.model.neuralnet import Neuralnet
from pypokerengine.engine.data_encoder import DataEncoder
from pypokerai.task import FOLD, CALL, MIN_RAISE, DOUBLE_RAISE, TRIPLE_RAISE, MAX_RAISE

MODEL_OUTPUT_ACTION_POSITION = [FOLD, CALL, MIN_RAISE, DOUBLE_RAISE, TRIPLE_RAISE, MAX_RAISE]
def action_index(action):
    return MODEL_OUTPUT_ACTION_POSITION.index(action["name"])

class RandomValueFunction(BaseApproxActionValueFunction):

    def __init__(self, blind_structure=None, handicappers=None):
        pass

    def setup(self):
        pass

    def construct_features(self, state, action):
        return state, action

    def approx_predict_value(self, features):
        return 0

    def approx_backup(self, features, backup_target, alpha):
        pass

class BasePokerActionValueFunction(BaseApproxActionValueFunction):

    MODEL_ARCHITECTURE_FILE_PATH = "model_architecture.json"
    MODEL_WEIGHTS_FILE_PATH = "model_weights.h5"

    def __init__(self, blind_structure, handicappers=None):
        self.blind_structure = blind_structure
        self.handicappers = handicappers
        self.prediction_cache = (None, None)  # (features, prediction)

    def setup(self):
        self.model = self.build_model()
        if not self.handicappers:
            self.handicappers = [Neuralnet("preflop"), Neuralnet("flop"), Neuralnet("turn"), Neuralnet("river")]
            [nn.compile() for nn in self.handicappers]

    def construct_features(self, state, action):
        my_uuid = state["table"].seats.players[state["next_player"]].uuid
        hole_card = [p for p in state["table"].seats.players if p.uuid==my_uuid][0].hole_card
        hole_str = [str(card) for card in hole_card]
        round_state = DataEncoder.encode_round_state(state)
        features = self.construct_poker_features(
                state, action, round_state, my_uuid, hole_str, self.handicappers, self.blind_structure)
        return features, action

    def approx_predict_value(self, features):
        X, action = features
        if self.prediction_cache[0] == X:
            values = self.prediction_cache[1]
        else:
            values = self.model.predict_on_batch(np.array([X]))[0].tolist()
            self.prediction_cache = (X, values)
        valur_for_action = values[action_index(action)]
        return valur_for_action

    def approx_backup(self, features, backup_target, alpha):
        X, action = features
        Y = self.model.predict_on_batch(np.array([X]))[0].tolist()
        Y[action_index(action)] = backup_target
        loss = self.model.train_on_batch(np.array([X]), np.array([Y]))
        self.prediction_cache = (None, None)

    def build_model(self):
        raise NotImplementedError("[build_model] method is not implemented")

    def construct_poker_features(
            self, state, action, round_staet, my_uuid, hole_str, handicappers, blind_structure):
        raise NotImplementedError("[construct_poker_features] method is not implemented")

    def visualize_feature_weights(self):
        raise NotImplementedError("[visualize_feature_weights] method is not implemented")

    def generate_features_title(self):
        raise NotImplementedError("[generate_features_title] method is not implemented")

    def save(self, save_dir_path):
        self.model.save_weights(os.path.join(save_dir_path, self.MODEL_WEIGHTS_FILE_PATH))

    def load(self, load_dir_path):
        self.model.load_weights(os.path.join(load_dir_path, self.MODEL_WEIGHTS_FILE_PATH))


class LinearModelScalarFeaturesValueFunction(BasePokerActionValueFunction):

    def build_model(self):
        input_dim = 35
        model = Sequential()
        model.add(Dense(6, input_dim=input_dim))
        model.compile(loss="mse",  optimizer="adam")
        return model

    def construct_poker_features(
            self, state, action, round_state, my_uuid, hole_str, handicappers, blind_structure):
        return F.construct_scalar_features(
                round_state, my_uuid, hole_str, blind_structure, neuralnets=handicappers)

    def visualize_feature_weights(self):
        return F.visualize_scalar_features_weight(self.model.get_weights())

    def generate_features_title(self):
        return F.scaled_scalar_features_title()

class LinearModelScaledScalarFeaturesValueFunction(BasePokerActionValueFunction):

    def build_model(self):
        input_dim = 35
        model = Sequential()
        model.add(Dense(6, input_dim=input_dim))
        model.compile(loss="mse",  optimizer="adam")
        return model

    def construct_poker_features(
            self, state, action, round_state, my_uuid, hole_str, handicappers, blind_structure):
        return F.construct_scaled_scalar_features(
                round_state, my_uuid, hole_str, blind_structure, neuralnets=handicappers)

    def visualize_feature_weights(self):
        return F.visualize_scaled_scalar_features_weight(self.model.get_weights())

    def generate_features_title(self):
        return F.scaled_scalar_features_title()

class LinearModelOnehotFeaturesValueFunction(BasePokerActionValueFunction):

    def build_model(self):
        input_dim = 83
        model = Sequential()
        model.add(Dense(6, input_dim=input_dim))
        model.compile(loss="mse",  optimizer="adam")
        return model

    def construct_poker_features(
            self, state, action, round_state, my_uuid, hole_str, handicappers, blind_structure):
        return F.construct_onehot_features(
                round_state, my_uuid, hole_str, blind_structure, neuralnets=handicappers)

    def visualize_feature_weights(self):
        return F.visualize_onehot_features_weight(self.model.get_weights())

    def generate_features_title(self):
        return F.onehot_features_title()

class LinearModelBinaryOnehotFeaturesValueFunction(BasePokerActionValueFunction):

    def build_model(self):
        input_dim = 281
        model = Sequential()
        model.add(Dense(6, input_dim=input_dim))
        model.compile(loss="mse",  optimizer="adam")
        return model

    def construct_poker_features(
            self, state, action, round_state, my_uuid, hole_str, handicappers, blind_structure):
        return F.construct_binary_onehot_features(
                round_state, my_uuid, hole_str, blind_structure, neuralnets=handicappers)

    def visualize_feature_weights(self):
        return F.visualize_binary_onehot_features_weight(self.model.get_weights())

    def generate_features_title(self):
        return F.binary_onehot_features_title()

