from keras.models import Model
from keras.layers import Input, GRU, RepeatVector, Conv1D, Add, Flatten, Dense
from tensorflow import keras
from keras.optimizers import Adam
import numpy as np
import tensorflow as tf
from keras import backend as K
import global_var as gol
from tensorflow.python.keras.regularizers import l2

def get_gtResNet(gru_units: int, conv_filters: int, conv_kernel_size: int, learning_rate: float,
                 input_shape: (int, int), output_units: int, type_quantile: bool = False) -> Model:
    inputs = Input(shape=input_shape)
    gru_output = GRU(gru_units, kernel_regularizer=l2(0.01))(inputs)
    dense_gru_output = Dense(conv_filters)(gru_output)
    repeated_output = RepeatVector(input_shape[0])(gru_output)
    conv_output = Conv1D(conv_filters, conv_kernel_size * 2, dilation_rate=conv_kernel_size,
                         activation='relu', padding='causal', kernel_regularizer=l2(0.01))(repeated_output)
    residual_output = Add()([dense_gru_output, conv_output])
    flattened_output = Flatten()(residual_output)
    outputs = Dense(output_units, activation='relu')(flattened_output)
    model = Model(inputs=inputs, outputs=outputs)
    optimizer = Adam(learning_rate=learning_rate)
    if type_quantile:
        model.compile(optimizer=optimizer, loss=quantile_loss)
    else:
        model.compile(optimizer=optimizer, loss='mse')
    # model.summary()
    return model
