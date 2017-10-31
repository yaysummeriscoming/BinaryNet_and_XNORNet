from keras.callbacks import Callback

from CustomLayers.XNORNetLayer import XNORNetConv2D
from CustomLayers.BinaryNetLayer import BinaryNetConv2D, BinaryNetActivation


# This file holds a dictionary of all custom layers, for use when loading a Keras model
customLayersDictionary = {
                          "XNORNetConv2D": XNORNetConv2D,
                          "BinaryNetActivation" : BinaryNetActivation,
                          "BinaryNetConv2D" : BinaryNetConv2D,
                          }


class CustomLayerUpdate(Callback):

    def on_batch_begin(self, batch, logs=None):
        for curLayer in self.model.layers:
            CallMethodName(object=curLayer, fn_name='on_batch_begin')

    def on_batch_end(self, batch, logs=None):
        for curLayer in self.model.layers:
            CallMethodName(object=curLayer, fn_name='on_batch_end')

    def on_epoch_begin(self, epoch, logs=None):
        for curLayer in self.model.layers:
            CallMethodName(object=curLayer, fn_name='on_epoch_begin')

    def on_epoch_end(self, epoch, logs=None):
        for curLayer in self.model.layers:
            CallMethodName(object=curLayer, fn_name='on_epoch_end')


# Call the desired class method if it exists
def CallMethodName(object, fn_name):
    fn = getattr(object, fn_name, None)
    if callable(fn):
        fn()

# Callbacks to implement custom layer specific code at the end of each training batch
customLayerCallbacks = [CustomLayerUpdate()]
