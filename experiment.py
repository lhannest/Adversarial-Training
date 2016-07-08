import theano.tensor.nnet as nnet
import theano.tensor as T
import theano
import numpy as np
from mnist import get_mnist

training_data, testing_data = get_mnist(100, 100)


def make_FC_layer(X, input_size, output_size, actfn=nnet.sigmoid):
    input_size += 1
    w_bound = np.sqrt(6. / (input_size - 1 + output_size))
    weights = np.random.uniform(-w_bound, w_bound, (input_size, output_size))
    # weights = np.random.randn(input_size, output_size)
    W = theano.shared(np.asarray(weights, dtype=theano.config.floatX))
    B = theano.shared(np.asarray([1], dtype=theano.config.floatX))
    X = T.concatenate([X, B])
    Z = T.dot(X, W)
    return actfn(Z), W

def make_model():
    Input = T.vector()
    Target = T.vector()
    Alpha = T.scalar()
    L1, W1 = make_FC_layer(Input, 784, 30)
    Output, W2 = make_FC_layer(L1, 30, 10)
    perams = [W1, W2]

    mse = T.sum(T.sqr(Output - Target))/2

    eval_model = theano.function(inputs=[Input], outputs=Output)
    train_model = theano.function(
        inputs=[Input, Target],
        outputs=mse,
        updates=[(w, w - T.grad(cost=mse, wrt=w)) for w in perams]
    )

    return eval_model, train_model

def mux(i):
    return [0.]*int(i) + [1.] + [0.]*(9-int(i))

def demux(array):
    return np.argmax(array)

def preprocess(img):
    img = np.array(img, dtype='float32')
    mean = np.mean(img)
    img -= mean
    return img

eval_model, train_model = make_model()

TOTAL_ITERATIONS = len(training_data)
ITERATIONS = 0
for img, label in training_data:
    ITERATIONS += 1
    label = mux(label)
    img = preprocess(img)
    error = train_model(img, label)
    if ITERATIONS % 10 == 0:
        print '\rerror:' + str(error) + '\t\t\t' + str(int(ITERATIONS * 100 / TOTAL_ITERATIONS)) + '% complete',
print ''
error = 0
for img, label in testing_data:
    img = preprocess(img)
    label = mux(label)
    y = eval_model(img)
    print demux(y), demux(label)
    if demux(y) != demux(label):
        error += 1
print 'error:', str(error) + '/' + str(len(testing_data)), str(error * 100. / len(testing_data)) + '%'
