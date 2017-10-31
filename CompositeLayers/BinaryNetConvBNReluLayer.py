from keras.layers import BatchNormalization, Activation
from CustomLayers.BinaryNetLayer import BinaryNetConv2D, BinaryNetActivation
from keras import backend as K

def BinaryNetConvBNReluLayer(input, nb_filters, border, kernel_size, stride, use_bias=True, data_format='channels_last', use_activation=False):
    output = input

    # BinaryNet uses binarisation as the activation
    # To get the graphs to compile properly, add binarisation as a seperate layer to the input for theano
    # The tensorflow implementation contains the input binarisation inside the layer definition
    if K.backend() == 'theano':
        output = BinaryNetActivation()(output)

    output = BinaryNetConv2D(nb_filters,
                             kernel_size,
                             use_bias=use_bias,
                             padding=border,
                             strides=stride,
                             data_format=data_format,
                             )(output)

    # Add output binarisation as a seperate layer for Theano
    if K.backend() == 'theano':
        output = BinaryNetActivation()(output)

    output = BatchNormalization()(output)

    if use_activation:
        output = Activation('relu')(output)

    return output