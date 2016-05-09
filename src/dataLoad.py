#Load the data 
import numpy
def loadData():
	f = open('/home/ashashantharam/Desktop/Columbia/BigDataAnalytics/Project/Data/fer2013/fer2013.csv','r')
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
	print(train_set_x.shape)
	print(train_set_y.shape)
	print(train_set_x[0])

if __name__ == "__main__":
	loadData()



