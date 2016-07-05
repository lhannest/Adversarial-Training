import theano
import theano.tensor as T
import theano.tensor.nnet as nnet
import numpy as np
from neuralnet import build_layer
import pylab
from PIL import Image
from utilities import imsave, OneHotEncoder
from generate_mnist_images import get_digits

encoder = OneHotEncoder(10)

w = np.load('mnist_weights.npy')
x = np.asarray(get_digits()[5]).reshape((28, 28))
imsave(x, 'images/5')

img = theano.shared(np.asarray(x.reshape((1, 784)) / 255., dtype=theano.config.floatX))
gen = theano.shared(np.asarray(np.random.randn(1, 784), dtype=theano.config.floatX))
A = gen + img
outputs1, weights1 = build_layer(A, input_size=784, output_size=300, weights=w[0])
outputs2, weights2 = build_layer(outputs1, input_size=300, output_size=10, activation=nnet.softmax, weights=w[1])

target = theano.shared(np.asarray([0, 0, 1, 0, 0, 0, 0, 0, 0, 0], dtype=theano.config.floatX))

# Regularize gen so that it doesn't get too big
cost = T.mean(T.sqr(outputs2 - target)) + T.mean(T.sqr(gen))

evaluate_model = theano.function(inputs=[], outputs=outputs2)
generate_image = theano.function(inputs=[], outputs=cost, updates=[(gen, gen - T.grad(cost, gen))])

for n in range(10):
    target.set_value(np.asarray(encoder.encode(n), dtype=theano.config.floatX))
    gen.set_value(np.asarray(np.random.randn(1, 784), dtype=theano.config.floatX))
    for i in range(10000):
        print i, generate_image()
    l = np.asarray(evaluate_model(), dtype='float')
    RESULT = encoder.decode(l)[0]

    a = encoder.maxval(l)
    filename = '{}{}{}{}'.format('images/adv_', RESULT, '_p', int(a*100))
    img = np.asarray(A.eval()).reshape((28, 28)) * 255
    imsave(img, filename)
