import keras.backend as K
import tensorflow as tf
from tensorflow.python.framework import function

def clipped_passthrough_grad_multiply(op, grad):
    return [K.clip(grad, -1., 1.), K.clip(grad, -1., 1.)]

def clipped_passthrough_grad(op, grad):
    return K.clip(grad, -1., 1.)


def variable_tanh_grad(op, grad):

    # tanh_grad = 1. - (grad * grad)
    #
    # return (tanh_grad + grad) / 2.

    tanh_grad = (1. - (op * op)) * grad

    return tanh_grad


def identity(op):
    return op

# COMEBACKTO_PBL
# Some defun examples:
# https://stackoverflow.com/questions/38833934/write-custom-python-based-gradient-function-for-an-operation-without-c-imple
# http://programtalk.com/python-examples/tensorflow.python.framework.function.Defun/
#
# https://stackoverflow.com/questions/39605798/treating-a-tensorflow-defun-as-a-closure






# @function.Defun(tf.float32, tf.float32, python_grad_func=clipped_passthrough_grad_multiply, func_name="passthroughMultiplyTF")
@function.Defun(tf.float32, tf.float32, func_name="passthroughMultiplyTF")
def passthroughMultiplyTF(x, y):
    x_new = tf.identity(x)
    y_new = tf.identity(y)
    output = x_new * y_new
    # output = tf.multiply(x_new, y_new)
    realOutput = tf.identity(output)

    return realOutput



# @function.Defun(tf.float32, python_grad_func=sign_grad, shape_func=identity, func_name="passthroughSign")
# @function.Defun(tf.float32, func_name="passthroughSign")
@function.Defun(tf.float32, python_grad_func=clipped_passthrough_grad, func_name="passthroughSignTF")
def passthroughSignTF(x):
    x_new = tf.identity(x)
    output = tf.sign(x_new)
    realOutput = tf.identity(output)

    return realOutput


@function.Defun(tf.float32, python_grad_func=clipped_passthrough_grad, func_name="passthroughTanhTF")
def passthroughTanhTF(x):
    x_new = tf.identity(x)
    output = tf.tanh(x_new)
    realOutput = tf.identity(output)

    return realOutput


def SetSession():
    print(tf.__version__)
    a = tf.Variable(tf.constant([-5., 4., -3., 2., 1.], dtype=tf.float32))

    # Make sure there's a reference to our custom passthroughSign function so that tensorflow includes it
    grad = tf.gradients(passthroughSignTF(a), [a])
    grad1 = tf.gradients(passthroughTanhTF(a), [a])
    grad2 = tf.gradients(passthroughMultiplyTF(a, a), [a])

    # COMEBACKTO_PBL: Testing multi-core usage
    jobs = 8

    config = tf.ConfigProto(intra_op_parallelism_threads=jobs, \
                            inter_op_parallelism_threads=jobs, \
                            allow_soft_placement=True, \
                            device_count={'CPU': jobs})

    # Set a new keras tensorflow session so that all of our custom tensorflow code is included
    sess = tf.Session(config=config)
    K.set_session(sess)