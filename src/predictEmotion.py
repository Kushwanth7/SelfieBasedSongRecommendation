#Accept the image and call loadModel.py and return the emotion
from PIL import Image
import numpy
from loadModel import loadModel
def predict(imageLocation="/home/ashashantharam/Desktop/sharan.png"):

	im = Image.open(imageLocation).convert("L")
	numpyArray = numpy.array(im)
	returnList = loadModel(imageArray = numpyArray.flatten())
	print(returnList)

if __name__ == "__main__":
    predict()
