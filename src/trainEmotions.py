#Code that trains the neural network for facial recognition
import numpy

import theano
import theano.tensor as T
from theano.tensor.signal import downsample

from hw3_nn import LogisticRegression, HiddenLayer, myMLP, LeNetConvPoolLayer, train_nn
from dataLoad import loadData
import cPickle

def test_emotionTraining(learning_rate=0.1, n_epochs=1000, nkerns=[16, 512, 20],
        batch_size=200, verbose=True):
    """
    Wrapper function for testing Multi-Stage ConvNet on SVHN dataset

    :type learning_rate: float
    :param learning_rate: learning rate used (factor for the stochastic
    gradient)

    :type n_epochs: int
    :param n_epochs: maximal number of epochs to run the optimizer

    :type nkerns: list of ints
    :param nkerns: number of kernels on each layer

    :type batch_size: int
    :param batch_szie: number of examples in minibatch.

    :type verbose: boolean
    :param verbose: to print out epoch summary or not to.

    """

    rng = numpy.random.RandomState(23455)

    datasets = loadData()

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0]
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
    n_test_batches = test_set_x.get_value(borrow=True).shape[0]
    n_train_batches //= batch_size
    n_valid_batches //= batch_size
    n_test_batches //= batch_size

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch

    x = T.matrix('x')   # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector of
                        # [int] labels

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    #Make learning rate a theano shared variable 
    learning_rate = theano.shared(learning_rate)
    print('... building the model')

    # Reshape matrix of rasterized images of shape (batch_size, 1 * 48 * 48)
    # to a 4D tensor, compatible with our LeNetConvPoolLayer
    layer0_input = x.reshape((batch_size, 1, 48, 48))
    
    

    # TODO: Construct the first convolutional pooling layer:
    layer0 = LeNetConvPoolLayer(
        rng,
        input=layer0_input,
        image_shape= (batch_size, 1, 48, 48),
        filter_shape= (nkerns[0],1,3,3),
        poolsize= (2,2)
    )

    # TODO: Construct the second convolutional pooling layer
    layer1 = LeNetConvPoolLayer(
        rng,
        input=layer0.output,
        image_shape= (batch_size, nkerns[0], 23, 23) ,
        filter_shape= (nkerns[1],nkerns[0],4,4),
        poolsize= (2,2)
    )

    # TODO: Construct the third convolutional pooling layer
    layer2 = LeNetConvPoolLayer(
        rng,
        input=layer1.output,
        image_shape= (batch_size,nkerns[1],10,10),
        filter_shape= (nkerns[2],nkerns[1],3,3),
        poolsize= (2,2)
    )

    # the HiddenLayer being fully-connected, it operates on 2D matrices of
    # shape (batch_size, num_pixels) (i.e matrix of rasterized images).
    # This will generate a matrix of shape (batch_size, nkerns[2] * 1 * 1).
    layer3_input = layer2.output.flatten(2)

    # construct a fully-connected sigmoidal layer
    layer3 = HiddenLayer(
        rng,
        input=layer3_input,
        n_in=nkerns[2] * 4 * 4,
        n_out= batch_size,
        activation=T.tanh
    )

    # classify the values of the fully-connected sigmoidal layer
    layer4 = LogisticRegression(input=layer3.output,
        n_in= batch_size,
        n_out=7)

    # the cost we minimize during training is the NLL of the model
    cost = layer4.negative_log_likelihood(y)

    # create a function to compute the mistakes that are made by the model
    test_model = theano.function(
        [index],
        layer4.errors(y),
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    validate_model = theano.function(
        [index],
        layer4.errors(y),
        givens={
            x: valid_set_x[index * batch_size: (index + 1) * batch_size],
            y: valid_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    # TODO: create a list of all model parameters to be fit by gradient descent
    params = layer0.params + layer1.params + layer2.params + layer3.params + layer4.params

    # create a list of gradients for all model parameters
    grads = T.grad(cost, params)

    # train_model is a function that updates the model parameters by
    # SGD Since this model has many parameters, it would be tedious to
    # manually create an update rule for each model parameter. We thus
    # create the updates list by automatically looping over all
    # (params[i], grads[i]) pairs.
    updates = [
        (param_i, param_i - learning_rate.get_value().item() * grad_i)
        for param_i, grad_i in zip(params, grads)
    ]

    train_model = theano.function(
        [index],
        cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )
    
    getPofYGivenX = theano.function(
        [index],
        layer4.pOfYGivenX(y),
        givens={
            x: valid_set_x[index * batch_size: (index + 1) * batch_size],
            y: valid_set_y[index * batch_size: (index + 1) * batch_size]
        }
    ) 

    ###############
    # TRAIN MODEL #
    ###############
    print('... training')

    train_nn(train_model, validate_model, test_model,
        n_train_batches, n_valid_batches, n_test_batches, n_epochs, learning_rate, verbose)
    print('Training the model complete')
    
    f1 = open('layer0', 'wb')
    cPickle.dump(layer0, f1, protocol=cPickle.HIGHEST_PROTOCOL)
    f1.close()
    
    f1 = open('layer1', 'wb')
    cPickle.dump(layer1, f1, protocol=cPickle.HIGHEST_PROTOCOL)
    f1.close()
    
    f1 = open('layer2', 'wb')
    cPickle.dump(layer2, f1, protocol=cPickle.HIGHEST_PROTOCOL)
    f1.close()

    f1 = open('layer3', 'wb')
    cPickle.dump(layer3, f1, protocol=cPickle.HIGHEST_PROTOCOL)
    f1.close()

    f1 = open('layer4', 'wb')
    cPickle.dump(layer4, f1, protocol=cPickle.HIGHEST_PROTOCOL)
    f1.close()

    print("Saving the model complete")
    
    predictedList = getPofYGivenX(1)
    
    print("List of probabilities predicted = " + str(predictedList))

if __name__ == "__main__":
	test_emotionTraining()
	