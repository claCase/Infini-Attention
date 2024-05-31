import tensorflow as tf
from tensorflow.keras import models
import numpy as np
#import pandas as pd
from tensorflow_probability.python.distributions import (
    RelaxedOneHotCategorical,
    OneHotCategorical,
    MultivariateNormalDiag,
    kl_divergence,
)
from src import layers

tfkl = tf.keras.layers


class InfiniAttentionModel(models.Model):
    def __init__(
        self,
        dims,
        heads,
        context_length,
        delta_rule=True,
        dropout=0,
        initializer="glorot_uniform",
        return_state=False,
        return_sequences=True,
        activation="tanh",
        positional_encoding=True,
        **kwargs
    ):
        super().__init__(**kwargs)
        cell = layers.AttentionRNNCell(
            dims=dims,
            heads=heads,
            context_length=context_length,
            delta_rule=delta_rule,
            dropout=dropout,
            initializer=initializer,
            positional_encoding=positional_encoding,
        )
        self.rnn = tfkl.RNN(
            cell, return_state=return_state, return_sequences=return_sequences
        )
        self.embedding = tfkl.Dense(dims*2, activation)
    
    def call(self, inputs, training):
        inputs = self.embedding(inputs)
        return self.rnn(inputs, training=training)
    