#Load the data 
import numpy
import theano
import theano.tensor as T

def shared_dataset(data_xy, borrow=True):
    """ Function that loads the dataset into shared variables

    The reason we store our dataset in shared variables is to allow
    Theano to copy it into the GPU memory (when code is run on GPU).
    Since copying data into the GPU is slow, copying a minibatch everytime
    is needed (the default behaviour if the data is not in a shared
    variable) would lead to a large decrease in performance.
    """
    data_x, data_y = data_xy
    shared_x = theano.shared(numpy.asarray(data_x,
                                           dtype=theano.config.floatX),
                             borrow=borrow)
    shared_y = theano.shared(numpy.asarray(data_y,
                                           dtype=theano.config.floatX),
                             borrow=borrow)
    # When storing data on the GPU it has to be stored as floats
    # therefore we will store the labels as ``floatX`` as well
    # (``shared_y`` does exactly that). But during our computations
    # we need them as ints (we use labels as index, and if they are
    # floats it doesn't make sense) therefore instead of returning
    # ``shared_y`` we will have to cast it to int. This little hack
    # lets ous get around this issue
    return shared_x, T.cast(shared_y, 'int32')

def loadData():
	f = open('/home/ubuntu/extern/SelfieBasedSongRecommendation/data/fer2013.csv','r')
	lines = f.read().split("\n")
	train_set_x = numpy.empty((28709,2304))
	valid_set_x = numpy.empty((3589,2304))
	test_set_x = numpy.empty((3589,2304))

	train_set_y = numpy.empty((28709,))
	valid_set_y = numpy.empty((3589,))
	test_set_y = numpy.empty((3589,))


	trainIndex = 0
	testIndex = 0
	validIndex = 0
	print(len(lines))
	for line in lines:
		values = line.split(',')
		arrX = numpy.fromstring(values[1],dtype=float,sep=' ')/255
		arrY = int(values[0])
		if(values[2] == 'Training'):
			train_set_x[trainIndex] = arrX.flatten()
			train_set_y[trainIndex] = arrY
			trainIndex = trainIndex + 1
		elif(values[2] == 'PublicTest'):
			valid_set_x[validIndex] = arrX.flatten()
			valid_set_y[validIndex] = arrY
			validIndex = validIndex + 1
		else:
			test_set_x[testIndex] = arrX.flatten()
			test_set_y[testIndex] = arrY
			testIndex = testIndex + 1

	#Make sets as theano shared variables
	train_set_x,train_set_y = shared_dataset([train_set_x,train_set_y])
	valid_set_x,valid_set_y = shared_dataset([valid_set_x,valid_set_y])
	test_set_x,test_set_y = shared_dataset([test_set_x,test_set_y])

	train_set = [train_set_x,train_set_y]
	valid_set = [valid_set_x,valid_set_y]
	test_set = [test_set_x,test_set_y]
	rval = [train_set, valid_set, test_set]
	return rval


if __name__ == "__main__":
	loadData()



