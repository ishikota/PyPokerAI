import pypokerai.features as F
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense
from kyoka.value_function import BaseApproxActionValueFunction
from holecardhandicapper.model.neuralnet import Neuralnet
from pypokerengine.engine.data_encoder import DataEncoder

class BasePokerActionValueFunction(BaseApproxActionValueFunction):

    def __init__(self, blind_structure):
        self.blind_structure = blind_structure

    def setup(self):
        self.model = self.build_model()
        self.handicappers = [Neuralnet("preflop"), Neuralnet("flop"), Neuralnet("turn"), Neuralnet("river")]
        [nn.compile() for nn in self.handicappers]

    def construct_features(self, state, action):
        my_uuid = state["table"].seats.players[state["next_player"]].uuid
        hole_card = [p for p in state["table"].seats.players if p.uuid==my_uuid][0].hole_card
        hole_str = [str(card) for card in hole_card]
        round_state = DataEncoder.encode_round_state(state)
        return self.construct_poker_features(
                state, action, round_state, my_uuid, hole_str, self.handicappers, self.blind_structure)

    def approx_predict_value(self, features):
        return self.model.predict_on_batch(np.array([features]))[0][0]

    def approx_backup(self, features, backup_target, alpha):
        loss = self.model.train_on_batch(np.array([features]), np.array([backup_target]))

    def build_model(self):
        raise NotImplementedError("[build_model] method is not implemented")

    def construct_poker_features(
            self, state, action, round_staet, my_uuid, hole_str, handicappers, blind_structure):
        raise NotImplementedError("[construct_poker_features] method is not implemented")

class LinearModelScalarFeaturesValueFunction(BasePokerActionValueFunction):

    def build_model(self):
        input_dim = 43
        model = Sequential()
        model.add(Dense(1, input_dim=input_dim))
        model.compile(loss="mse",  optimizer="adam")
        return model

    def construct_poker_features(
            self, state, action, round_state, my_uuid, hole_str, handicappers, blind_structure):
        return F.construct_scalar_features(
                round_state, my_uuid, hole_str, blind_structure, action, neuralnets=handicappers)

class LinearModelScaledScalarFeaturesValueFunction(BasePokerActionValueFunction):

    def build_model(self):
        input_dim = 43
        model = Sequential()
        model.add(Dense(1, input_dim=input_dim))
        model.compile(loss="mse",  optimizer="adam")
        return model

    def construct_poker_features(
            self, state, action, round_state, my_uuid, hole_str, handicappers, blind_structure):
        return F.construct_scaled_scalar_features(
                round_state, my_uuid, hole_str, blind_structure, action, neuralnets=handicappers)

class LinearModelOnehotFeaturesValueFunction(BasePokerActionValueFunction):

    def build_model(self):
        input_dim = 109
        model = Sequential()
        model.add(Dense(1, input_dim=input_dim))
        model.compile(loss="mse",  optimizer="adam")
        return model

    def construct_poker_features(
            self, state, action, round_state, my_uuid, hole_str, handicappers, blind_structure):
        return F.construct_onehot_features(
                round_state, my_uuid, hole_str, blind_structure, action, neuralnets=handicappers)
