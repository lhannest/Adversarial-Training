import theano
import theano.tensor as T
import theano.tensor.nnet as nnet
import numpy as np

def build_neural_net(X, filename=None):
    """Builds the Theano symbolic neural network"""
    Y1, weights1 = build_layer(X, input_size=784, output_size=300)
    Y2, weights2 = build_layer(Y1, input_size=300, output_size=10, activation=nnet.softmax)

    if filename != None:
        saved_weights = np.load(filename)
        weights1.set_value(np.asarray(saved_weights[0], dtype=theano.config.floatX))
        weights2.set_value(np.asarray(saved_weights[1], dtype=theano.config.floatX))

    return Y2, weights1, weights2

def build_mnist_model(filename=None):
    """Builds functions with which to train, evaluate and save the neural network"""
    inputs = T.matrix()
    targets = T.matrix()
    alpha = T.scalar()

    outputs, weights1, weights2 = build_neural_net(inputs, filename)
    params = [weights1, weights2]

    mse = T.mean(T.sqr(outputs - targets))
    updates = [(p, p - alpha * T.grad(cost=mse, wrt=p)) for p in params]

    evaluate_model = theano.function(inputs=[inputs], outputs=outputs, allow_input_downcast=True)
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

def build_layer(X, input_size, output_size, activation=nnet.sigmoid):
    r = np.sqrt(6. / (input_size + output_size))
    w = np.random.uniform(-r, r, (input_size + 1, output_size))
    W = theano.shared(np.asarray(w, dtype=theano.config.floatX))
    # Here we force X to be a 2d array. If X is already a 2d array, it will be unchanged
    X = X.reshape((-1, X.shape[-1]))
    # Here we append a column of ones, for the bias inputs of this layer
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
