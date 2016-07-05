from sklearn.datasets import fetch_mldata
import numpy as np
from PIL import Image

def get_mnist_data(training_size, testing_size):
    msg = "There are only 70,000 MNIST datapoints"
    assert training_size + testing_size <= 70000, msg
    mnist = fetch_mldata('MNIST original', data_home='./data')
    data = zip(mnist.data / 255., mnist.target)
    np.random.shuffle(data)
    training_data = data[0:training_size]
    testing_data = data[training_size:training_size+testing_size]
    return training_data, testing_data

def get_digits():
    data, _ = get_mnist_data(70000, 0)

    imgs = {}
    for x, t in data:
        imgs[int(t)] = np.asarray(x).reshape((28, 28)) * 255

        if len(imgs) == 10:
            break
    return [imgs[i] for i in range(10)]

if __name__ == '__main__':
    from utilities import imsave
    for i, img in enumerate(get_digits()):
        imsave(img, 'images/mnist_' + str(i))
