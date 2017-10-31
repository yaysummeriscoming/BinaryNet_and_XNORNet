import keras.backend as K

if K.backend() == 'tensorflow':

    import CustomOps.tensorflowOps as tensorflowOps

    def passthroughSign(x):
        return tensorflowOps.passthroughSignTF(x)

    def passthroughTanh(x):
        return tensorflowOps.passthroughTanhTF(x)

    def SetSession():
        tensorflowOps.SetSession()

elif K.backend() == 'theano':

    from CustomOps.theanoOps import BinaryTanh

    def passthroughSign(x):
        return BinaryTanh(x)

    def passtrhoughTanh(x):
        assert "This op hasn't been programmed for theano yet"

    def SetSession():
        empty = True


else:
    raise NameError('backend not supported')