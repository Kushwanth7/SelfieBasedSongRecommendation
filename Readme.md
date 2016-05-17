#SONG RECOMMENDATION BASED ON SELFIE

PROJECT FOR EECS 6895 ADVANCED BIG DATA ANALYTICS 
```Text
Contributors-
Kushwanth Ram - kk3098, github: kushwanth7
Sharan Suryanarayanan - ss4951, github: s-sharan
```

##ABSTRACT
Mood based song recommendations have existed for a long time, but in majority of the scenarios the mood is determined after learning the user preferences over a period of time, like looking at his past song preferences, time he listens to the music etc. Some of the mood based song recommendation systems like [Smoodi](http://ieeexplore.ieee.org/xpls/abs_all.jsp?arnumber=6012116&tag=1
) classify music based on mood and later make song recommendations to the user based on his selection of the mood, this and many other mood based music recommendation systems work by asking the user to select a mood and song recommendations are made based on the mood selected manually by the user. In this paper we propose a new approach to mood based song recommendation, where in the mood of a person is determined from his picture and based on the mood predicted song recommendations are made that best suit the mood predicted.

##DESCRIPTION
We have a built a website making use of HTML, Angular JS, Bootstrap which allows an user to take a selfie. We make use of Python Flask as a web server to host this web page and also to act as the backend system. When the user takes a selfie the image data is transmitted to the backend system, and this image is passed through the trained neural network model to determine the mood and then based on the mood predicted songs suitable for this mood are queried from database and one of the song is returned to the webpage along with the predicted mood. We then play this song on the web page by making use of the YouTube API.

So we are open- sourcing our code for the website, python flask web server, the neural-network model that we have saved. We are also open sourcing our theano code that we wrote to implement the neural network. Also our repository contains the open-source image data from Kaggle that we used to train our model. We have made use of an Apache Open source license for our project.

Our repository contains two folders, the data folder where the training data is stored and the src folder which contains all the codes. The templates folder contains the code for html webpage which allows the user to interact with the application. This where the user can capture a selfie and some song is recommended to him based on his mood. 
To run the web server install Python Flask on your system and the run the app.py file. The app.py file contains the logic for the web server and it acts as the backend system of our application interacting with the user, databases and the neural network model. The songs are stored in  DynamoDB on Amazon AWS, we are not open sourcing this information. We have an unique story in the way we recommend the songs for each mood, for example when other sites recommend sad songs when a person is sad or feeling bad, we recommend users with songs which will cheer them up. The code to train the neural network can be found in the Train Emotion Recognizer iPython notebook. If anyone wants to modify the network to suit their particular needs or feel it is necessary to tweak the network they can do so by making changes to the code present there. 

Thus the user can implement our project by cloning our repository and creating a DynamoDB with the song information and running our code. If they do not want to make use of DynamoDB, then they will have to tweak the code to suit their needs.


##Data
We have obtained the dataset of around 40000 pre-classified images from Kaggle's Learn facial expressions from an image. Each of these images were classified as to belonging to one of Angry, Disgust, Fear, Happy, Sad, Surprise or Neutral emotional categories. 3589 images from the same dataset was used as test dataset and another 3589 images was used as part of the validation set during training. Each of the images in the dataset were 48 * 48 grey scale images. 

[Kaggle's learn Facial Expression dataset] (https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge)

##Tools
1. Theano
2. Python Flask
3. AngularJS
4. Amazon DynamoDB
5. YouTube API
