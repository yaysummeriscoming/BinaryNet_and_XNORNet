from keras.layers import Activation, BatchNormalization
from keras.layers import Convolution2D

def ConvBNReluLayer(input, nb_filters, border, kernel_size, stride, use_bias=True, data_format='channels_last'):

        output = Convolution2D(filters=nb_filters,
                               kernel_size=kernel_size,
                               strides=stride,
                               padding=border,
                               data_format=data_format,
                               use_bias=use_bias
                               )(input)

        output = BatchNormalization()(output)
        output = Activation('relu')(output)

        return output
