import keras.backend as K
import numpy as np
from keras.engine import InputSpec
from keras.engine import Layer
from keras.layers import Convolution2D

from CustomOps.customOps import passthroughSign


class BinaryNetActivation(Layer):

    def __init__(self, **kwargs):
        super(BinaryNetActivation, self).__init__(**kwargs)
        # self.supports_masking = True

    def build(self, input_shape):
        super(BinaryNetActivation, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, inputs):
        # In BinaryNet, the output activation is binarised (normally done at the input to each layer in our implementation)
        return passthroughSign(inputs)

    def get_config(self):
        base_config = super(BinaryNetActivation, self).get_config()
        return base_config

    def compute_output_shape(self, input_shape):
        return input_shape

class BinaryNetConv2D(Convolution2D):
    """2D binary convolution layer (e.g. spatial convolution over images).

    This is an implementation of the BinaryNet layer described in:
    Binarized Neural Networks: Training Deep Neural Networks with Weights and Activations Constrained to +1 or -1

    It's based off the Convolution2D class, featuring an idential argument list.

    NOTE: The weight binarisation functionality is implemented using a 'on batch end' function,
    which must be called at the end of every batch (ideally using a callback).  Currently this functionality
    is implemented using Numpy.  In practice this incurs a negligible performance penalty,
    as this function uses far fewer operations than the base convolution operation.

    # Arguments
        Same as base Convolution2D layer

    # Input shape
        4D tensor with shape:
        `(samples, channels, rows, cols)` if data_format='channels_first'
        or 4D tensor with shape:
        `(samples, rows, cols, channels)` if data_format='channels_last'.

    # Output shape
        4D tensor with shape:
        `(samples, filters, new_rows, new_cols)` if data_format='channels_first'
        or 4D tensor with shape:
        `(samples, new_rows, new_cols, filters)` if data_format='channels_last'.
        `rows` and `cols` values might have changed due to padding.
    """

    def build(self, input_shape):
        # Call the build function of the base class (in this case, convolution)
        # super(BinaryNetConv2D, self).build(input_shape)  # Be sure to call this somewhere!
        super().build(input_shape)  # Be sure to call this somewhere!

        # Get the initialised weights as save as the 'full precision' weights
        weights = K.get_value(self.weights[0])
        self.fullPrecisionWeights = weights.copy()

        # Compute the binary approximated weights & save ready for the first batch
        B = np.sign(self.fullPrecisionWeights)
        self.lastIterationWeights = B.copy()
        K.set_value(self.weights[0], B)


    def call(self, inputs):

        # For theano, binarisation is done as a seperate layer
        if K.backend() == 'tensorflow':
            binarisedInput = passthroughSign(inputs)
        else:
            binarisedInput = inputs

        return super().call(binarisedInput)


    def on_batch_end(self):
        # Weight arrangement is: (kernel_size, kernel_size, num_input_channels, num_output_channels)
        # for both data formats in keras 2 notation

        # Work out the weights update from the last batch and then apply this to the full precision weights
        # The current weights correspond to the binarised weights + last batch update
        newWeights = K.get_value(self.weights[0])
        weightsUpdate = newWeights - self.lastIterationWeights
        self.fullPrecisionWeights += weightsUpdate
        self.fullPrecisionWeights = np.clip(self.fullPrecisionWeights, -1., 1.)

        # Work out new approximated weights based off the full precision values
        B = np.sign(self.fullPrecisionWeights)

        # Save the weights, both in the keras kernel and a reference variable
        # so that we can compute the weights update that keras makes
        self.lastIterationWeights = B.copy()
        K.set_value(self.weights[0], B)
