import theano.tensor.nnet as nnet
import theano.tensor as T
import theano
import numpy as np
from mnist import get_mnist
import matplotlib.pyplot as plt

def mux(i):
    i = int(i)
    x = [0 for j in range(10)]
    x[i] = 1
    return x

def demux(array):
    return int(np.argmax(array))

def crossEntropyError(Y, Y_hat):
    return -T.mean(T.log(Y)[T.arange(Y_hat.shape[0]), Y_hat])

# training_data, testing_data = get_mnist(1000, 100)
training_data = [
                    ([0, 0], [0]),
                    ([0, 1], [1]),
                    ([1, 0], [1]),
                    ([1, 1], [0]),
                ]
testing_data = training_data
training_data = training_data * 10

X = T.vector()
t = T.vector()
W1 = theano.shared(np.asarray(np.random.randn(2, 8), dtype=theano.config.floatX))
W2 = theano.shared(np.asarray(np.random.randn(8, 1), dtype=theano.config.floatX))
perams = [W1, W2]
A = nnet.sigmoid(T.dot(X, W1))
Y = nnet.softmax(T.dot(A, W2))
mse = T.mean(T.sqr(Y - t))

train_model = theano.function(inputs=[X, t], outputs=mse, updates=[(p, p - 0.00000001*T.grad(mse, p)) for p in perams], allow_input_downcast=True)
eval_model = theano.function(inputs=[X], outputs=Y, allow_input_downcast=True)
for img, label in training_data:
    # img = img - np.mean(img)
    # label = mux(label)
    print 'err:', train_model(img, label)

success = 0
for img, label in testing_data:
    # img = img - np.mean(img)
    y = eval_model(img)
    # print y, demux(y), label
    # if demux(y) == label:
    if y == label:
        success += 1

print success * 100. / len(testing_data), '% accuracy'
