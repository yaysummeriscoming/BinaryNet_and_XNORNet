This project contains Keras implementations of the BinaryNet and XNORNet papers:

[Binarized Neural Networks: Training Deep Neural Networks with Weights and Activations Constrained to +1 or -1](https://arxiv.org/abs/1602.02830)


[XNOR-Net: ImageNet Classification Using Binary Convolutional Neural Networks](https://arxiv.org/abs/1603.05279)

Code supports the Tensorflow and Theano backends.

The most difficult part of coding these implementations was the sign function gradient.  I’ve used the clipped ‘passthrough’ sign implementation detailed in the BinaryNet paper.  The XNORNet doesn’t mention anything, so I’ve used the same implementation here too.
