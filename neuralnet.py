import theano
import theano.tensor as T
import theano.tensor.nnet as nnet
import numpy as np

def build_mnist_model(filename=None):
    inputs = T.matrix()
    targets = T.matrix()
    alpha = T.scalar()

    w1 = None
    w2 = None
    if filename != None:
        w = np.load(filename)
        w1 = w[0]
        w2 = w[1]

    outputs1, weights1 = build_layer(inputs, input_size=784, output_size=300, weights=w1)
    outputs2, weights2 = build_layer(outputs1, input_size=300, output_size=10, activation=nnet.softmax, weights=w2)
    params = [weights1, weights2]

    mse = T.mean(T.sqr(outputs2 - targets))
    updates = [(p, p - alpha * T.grad(cost=mse, wrt=p)) for p in params]

    evaluate_model = theano.function(inputs=[inputs], outputs=outputs2, allow_input_downcast=True)
    train_model = theano.function(inputs=[inputs, targets, alpha], outputs=mse, updates=updates, allow_input_downcast=True)

    def save_model(filename):
        w1, w2 = np.asarray(weights1.eval()), np.asarray(weights2.eval())
        w = np.asarray([w1, w2])
        np.save(filename, w)

    return evaluate_model, train_model, save_model

def build_xor_model():
    inputs = T.matrix()
    targets = T.matrix()
    alpha = T.scalar()
    outputs1, weights1 = build_layer(inputs, input_size=2, output_size=4)
    outputs2, weights2 = build_layer(outputs1, input_size=4, output_size=1)
    params = [weights1, weights2]

    mse = T.mean(T.sqr(outputs2 - targets))
    updates = [(p, p - alpha * T.grad(cost=mse, wrt=p)) for p in params]

    evaluate = theano.function(inputs=[inputs], outputs=outputs2)
    train = theano.function(inputs=[inputs, targets, alpha], outputs=mse, updates=updates)

    def evaluate_model(x):
        if len(x.shape) < 2:
            np.reshape(x, (1,) + x.shape)
        return evaluate(x)

    return evaluate_model, train_model

def build_layer(X, input_size, output_size, activation=nnet.sigmoid, weights=None):
    r = np.sqrt(6. / (input_size + output_size))
    if weights is None:
        w = np.random.uniform(-r, r, (input_size + 1, output_size))
    else:
        w = weights
    W = theano.shared(np.asarray(w, dtype=theano.config.floatX))
    X = T.concatenate([X, T.ones((X.shape[0], 1))], axis=1)
    Z = T.dot(X, W)
    Y = activation(Z)
    return Y, W


if __name__ == '__main__':
    evaluate_model, train_model = build__xor_model()

    inputs = [[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]]

    targets = [[0], [1], [1], [0]]

    for i in range(1000):
        print train_model(inputs, targets, 10)

    for x, t in zip(inputs, targets):
        print x, evaluate_model([x]), t
