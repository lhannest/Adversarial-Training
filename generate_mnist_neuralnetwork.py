import theano
import theano.tensor as T
import theano.tensor.nnet as nnet
from sklearn.datasets import fetch_mldata
import numpy as np
from neuralnet import build_mnist_model
from utilities import Printer, unzip, OneHotEncoder
import pylab

def get_mnist_data(training_size, testing_size):
    msg = "There are only 70,000 MNIST datapoints"
    assert training_size + testing_size <= 70000, msg
    mnist = fetch_mldata('MNIST original', data_home='./data')
    data = zip(mnist.data / 255., mnist.target)
    np.random.shuffle(data)
    training_data = data[0:training_size]
    testing_data = data[training_size:training_size+testing_size]
    return training_data, testing_data

training_data, testing_data = get_mnist_data(60000, 10000)
evaluate, train, save = build_mnist_model()

printer = Printer(0.1)
encoder = OneHotEncoder(10)

iterations = 5
batch_sizes = 5
for itr in range(iterations):
    # Preparing the training data
    np.random.shuffle(training_data)
    inputs, targets = unzip(training_data)
    input_batches = np.array_split(np.asarray(inputs), len(inputs) / batch_sizes)
    target_batches = np.array_split(np.asarray(targets), len(targets) / batch_sizes)

    if itr == 0:
        print 'training with', len(input_batches), 'batches of size', len(input_batches[0])
        total_iteartions = len(input_batches) * iterations

    for i, (input_batch, target_batch) in enumerate(zip(input_batches, target_batches)):
        target_batch = encoder.encode(target_batch)
        error = train(input_batch, target_batch, 4)
        current_iteration = itr*len(input_batches) + i
        printer.overwrite('training ' + str(int(current_iteration * 100. / total_iteartions)) + '% - error:' + str(error))

printer.clear()

error = 0
for i, (x, t) in enumerate(testing_data):
    printer.overwrite('testing on testing_data ' + str(i * 100 / len(testing_data)) + '%')
    y = evaluate([x])
    if encoder.decode(y) != t:
        error += 1

printer.clear()

print 'test error:', str(error * 100. / len(testing_data)) + '%'

error = 0
for i, (x, t) in enumerate(training_data):
    printer.overwrite('testing on training_data ' + str(i * 100 / len(training_data)) + '%')
    y = evaluate([x])
    if encoder.decode(y) != t:
        error += 1

printer.clear()

print 'training error:', str(error * 100. / len(training_data)) + '%'

save('mnist_weights')
