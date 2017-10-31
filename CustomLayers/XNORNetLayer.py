import keras.backend as K
from keras.layers import Convolution2D
from keras.engine import InputSpec
import numpy as np

from CustomOps.customOps import passthroughSign

class XNORNetConv2D(Convolution2D):
    """2D 'XNORNet' convolution layer (e.g. spatial convolution over images).

    This is an implementation of the XNORNet layer described in:
    XNOR-Net: ImageNet Classification Using Binary Convolutional Neural Networks

    It's based off the Convolution2D class, featuring an identical argument list, with the addition of
    a 'binarise input' parameter.

    NOTE: The weight binarisation functionality is implemented using a 'on batch end' function,
    which must be called at the end of every batch (ideally using a callback).  Currently this functionality
    is implemented using Numpy.  In practice this incurs a negligible performance penalty,
    as this function uses far fewer operations than the base convolution operation.

    # Arguments
        Same as base Convolution2D layer, except:
        binarise_input: This controls whether we operate with just binary weights, or with binarised activations as well

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

    def __init__(self,
                 binarise_input=True,
                 **kwargs):

        super().__init__(**kwargs)
        self.input_spec = InputSpec(ndim=4)

        self.binarise_input = binarise_input


    def build(self, input_shape):
        # Call the build function of the base class (in this case, convolution)
        super(XNORNetConv2D, self).build(input_shape)  # Be sure to call this somewhere!

        # k filter should be of shape (filter_size, filter_size, 1, 1), following standard keras 2 notation
        k_numpy = np.ones(shape=(self.kernel_size[0], self.kernel_size[1], 1, 1))
        k_numpy = k_numpy / np.sum(k_numpy)
        self.k_filter = K.variable(k_numpy, dtype='float32')

        weights = K.get_value(self.weights[0])
        self.fullPrecisionWeights = weights.copy()

        B = np.sign(self.fullPrecisionWeights)

        # Calculate a seperate alpha value for each filter
        alpha = np.mean(np.abs(self.fullPrecisionWeights), axis=(0, 1, 2))
        alphaB = np.broadcast_to(alpha, B.shape)

        newApproximatedWeights = np.multiply(alphaB, B)
        self.lastIterationWeights = newApproximatedWeights.copy()
        K.set_value(self.weights[0], newApproximatedWeights)


    def call(self, inputs):
        # Channels first arrangement: (batch_size, num_input_channels, width, height)
        # Channels last arrangement: (batch_size, width, height, num_input_channels)

        # If activation quantisation is enabled
        if self.binarise_input:

            # Compute the axis ID of the channels.  Use tensorflow channels last arrangement as standard
            channels_axis = 3

            if self.data_format == 'channels_first':
                channels_axis = 1

            # Compute A, which is the average across channels.
            # The input will thus reduce to a single-channel image
            # In Keras, (minibatch_size, 1, height, width)
            A = K.mean(K.abs(inputs), axis=channels_axis, keepdims=True)

            # k filter should be of shape (filter_size, filter_size, 1, 1) as per keras 2 notation
            # K is of shape (batch_size, 1, width, height) (using channels first data format)
            K_variable = K.conv2d(A,
                                  self.k_filter,
                                  strides=self.strides,
                                  padding=self.padding,
                                  data_format=self.data_format,
                                  dilation_rate=self.dilation_rate)

            # Binarise the input
            binarisedInput = passthroughSign(inputs)

            # Call the base convolution operation
            # Convolution output will be of shape (batch_size, width, height, num_output_channels) (channels first)
            convolutionOutput = K.conv2d(
                binarisedInput,
                self.kernel,
                strides=self.strides,
                padding=self.padding,
                data_format=self.data_format,
                dilation_rate=self.dilation_rate)

            # Copy K for each output channel
            # K will thus go from shape (batch_size, 1, width, height) to (batch_size, 1, width, height)
            # (with channels_first data format)
            if K.backend() == 'tensorflow':
                K_variable = K.repeat_elements(K_variable, K.int_shape(convolutionOutput)[channels_axis], axis=channels_axis)
            else:
                K_variable = K.repeat_elements(K_variable, K.shape(convolutionOutput)[channels_axis], axis=channels_axis)

            outputs = K_variable * convolutionOutput

            return outputs

        else:
            # Call the base convolution operation.  Only the weights are quantised in this case
            return super(XNORNetConv2D, self).call(inputs)


    def on_batch_end(self):
        # Weight arrangement is: (kernel_size, kernel_size, num_input_channels, num_output_channels)
        # for both data formats in keras 2 notation

        # Work out the weights update from the last batch and then apply this to the full precision weights
        # The current weights correspond to the binarised weights + last batch update
        newWeights = K.get_value(self.weights[0])
        weightsUpdate = newWeights - self.lastIterationWeights
        self.fullPrecisionWeights += weightsUpdate

        # Calculate the binary 'B' and 'alpha' scaling factors for each filter
        B = np.sign(self.fullPrecisionWeights)
        alpha = np.mean(np.abs(self.fullPrecisionWeights), axis=(0, 1, 2))
        alphaB = np.broadcast_to(alpha, B.shape)

        # Save the weights, both in the keras kernel and a reference variable
        # so that we can compute the weights update that keras makes
        newApproximatedWeights = np.multiply(alphaB, B)
        self.lastIterationWeights = newApproximatedWeights.copy()

        K.set_value(self.weights[0], newApproximatedWeights)
