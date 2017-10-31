from keras.layers import Activation, BatchNormalization
from CustomLayers.XNORNetLayer import XNORNetConv2D

def BNXNORConvReluLayer(input,
                        nb_filters,
                        border,
                        kernel_size,
                        stride,
                        use_BN=True,
                        use_bias=False,
                        use_activation=True,
                        binarise_input=True,
                        data_format='channels_last'):

    output = input

    if use_BN:
        output = BatchNormalization()(output)

    output = XNORNetConv2D(filters=nb_filters,
                           kernel_size=kernel_size,
                           use_bias=use_bias,
                           padding=border,
                           strides=stride,
                           data_format=data_format,
                           binarise_input=binarise_input
                           )(output)

    if use_activation:
        output = Activation('relu')(output)

    return output


def XNORConvBNReluLayer(input,
                        nb_filters,
                        border,
                        kernel_size,
                        stride,
                        use_BN=True,
                        use_bias=False,
                        use_activation=True,
                        binarise_input=True,
                        data_format='channels_last'):

    output = input

    output = XNORNetConv2D(nb_filters=nb_filters,
                           kernel_size=kernel_size,
                           use_bias=use_bias,
                           padding=border,
                           strides=stride,
                           data_format=data_format,
                           binarise_input=binarise_input
                           )(output)

    if use_BN:
        output = BatchNormalization()(output)

    if use_activation:
        output = Activation('relu')(output)

    return output
