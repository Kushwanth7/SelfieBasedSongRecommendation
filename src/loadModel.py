#Load the trained model and accept as parameter an image for which the emotion has to be predicted
import numpy

import theano
import theano.tensor as T
from theano.tensor.signal import downsample

from hw3_nn import LogisticRegression, HiddenLayer, myMLP, LeNetConvPoolLayer, train_nn
from dataLoad import loadData
import cPickle
def loadModel(image):

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

    
    getPofYGivenX = theano.function(
        [index],
        layer4.pOfYGivenX(),
        givens={
            x: valid_set_x[index: (index + 1)],
            y: valid_set_y[index: (index + 1)]
        },
        on_unused_input='ignore'
    )

    #Load the saved parameters
    f = open('savedParameters','rb')
	params = cPickle.load(f)     
    f.close()

    layer0.W = params[0]
    layer0.b = params[1]
    layer1.W = params[2]
    layer1.b= params[3]
    layer2.W = params[4]
    layer2.b = params[5]
    layer3.W = params[6]
    layer3.b = params[7]
    layer4.W = params[8]
    layer4.b = params[9]

    predictedList = getPofYGivenX(1)
    
    print("List of probabilities predicted = " + str(predictedList))
