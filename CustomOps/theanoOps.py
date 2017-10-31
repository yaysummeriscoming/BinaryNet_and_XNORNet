from keras.backend import theano_backend as T
from theano.scalar.basic import UnaryScalarOp, same_out_nocomplex
from theano.tensor.elemwise import Elemwise

# Our own rounding function that does not set the gradient to 0 like Theano's
class __Round(UnaryScalarOp):

    def c_code(self, node, name, inputs, outputs, sub):
        x, = inputs
        z, = outputs
        return "%(z)s = round(%(x)s);" % locals()

    def grad(self, inputs, gout):
        (gz,) = gout
        return gz,

__round_scalar = __Round(same_out_nocomplex, name='__round')
__round = Elemwise(__round_scalar)

def HardSigmoid(x):
    return T.clip((x+1.)/2.,0,1)

# The neurons' activations binarization function
# It behaves like the sign function during forward propagation
# And like:
#   hard_tanh(x) = 2*hard_sigmoid(x)-1.
# during back propagation
def BinaryTanh(x):
    return 2.*__round(HardSigmoid(x))-1.