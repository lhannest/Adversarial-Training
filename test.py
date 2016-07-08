import theano.tensor.nnet as nnet
import theano.tensor as T
import theano
import numpy as np

isize = 3
osize = 2

x = np.random.randn(isize)
w = np.random.randn(isize, osize)
t = np.random.randn(osize)
X = theano.shared(np.asarray(x, dtype=theano.config.floatX))
W = theano.shared(np.asarray(w, dtype=theano.config.floatX))
_Y = theano.shared(np.asarray(t, dtype=theano.config.floatX))
Z = T.dot(X, W)
Y = nnet.softmax(Z)
E = Y - _Y

print X.eval()
print W.eval()
print Y.eval()
print _Y.eval()
print E.eval()
print T.mean(E).eval()
