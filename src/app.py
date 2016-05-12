from flask import Flask, render_template, request, redirect, url_for
import json
import base64
from PIL import Image
import os
from loadModel import loadModel
import numpy
from dataLoad import loadData, shared_dataset
app = Flask(__name__)

datasets = loadData(shared=False)
validSets = datasets[1]
@app.route('/')
def index():
	return render_template('index.html')


@app.route("/pic",methods=['POST'])
def picture():
	modes = ['Angry','Disgust','Fear','Happy','Sad','Surprise','Neutral', 'Could not Predict emotion']
	modeId = 7
	if(request.method== 'POST'):
		content = request.get_json()["data"]
		data = content.split(',')[1].decode("base64")
		file1=open('pic.png','wb')
		file1.write(data)
		file1.close()
		img = Image.open('pic.png').convert('L')
		numpyArray = numpy.array(img)
		returnList = loadModel(imageArray = numpyArray.flatten(), validSet = validSets)
		os.remove('pic.png')
		modeId = returnList[0]

	return str(json.dumps(modes[modeId]))

def save():
	return render_template('index2.html')
if __name__ == "__main__":
	 app.run()