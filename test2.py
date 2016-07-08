import theano
import theano.tensor as T
import theano.tensor.nnet as nnet
import numpy as np

x = np.random.randn(2)
w = np.random.randn(3, 1)
x = theano.shared(np.asarray(x, dtype=theano.config.floatX))

a = T.concatenate([x, T.ones((x.shape[0], 1))], axis=1)
print np.ones((3, 1))
print x
print a.eval()
print 'dot:'
print T.dot(a, w).eval()
