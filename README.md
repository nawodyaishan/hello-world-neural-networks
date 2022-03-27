---

Introduction to Machine Learning and Hello World in Neural Networks

---

Hi fellows,
Here we are going to build basic pattern recognition models using neural networks. Neural Networks are a more advanced topic in machine learning that belongs to deep learning butin this basic introductory example we use the external libraries to make things easy a little bit.
First of you have to have some basics prerequisites of machine learning
What is Machine Learning?
Yes, as the topic says machines are learning. Machine learning is all about how machines are learning and how do we teach the machines to learn. In so near future machines can learn and do the thing that we, humans do.
From the computer programming perspective, traditionally we write computer programs by defining rules and data by ourselves to the computer. As the following diagram, the user gives rules and data and answers come out.
But in a machine learning scenario, we give answers and data to the machine, the computer and it produces the rules, the connection between answers, and data itself. Basically, the thing happening here is we give the exiting examples to the computer which we cant figure out how it's happening, and through machine recognizes the how it happening by the rules.
Let's take a quick simple example,
Here I show you, a Human coded BMI Calculator using Python.
# height input as "h" and weight in as "w"
h = float(input("Please Enter Your Height in cm: "))
w = float(input("Please Enter Your Weight in kg: "))
BMI = w / (h/100)**2
As above code, we can calculate each person's BMI index and give feedback on their inputs as follows.
print(f"You BMI Index is {BMI}")
if BMI <= 18.4:
    print("You are slightly underweight.")
elif BMI <= 24.9:
    print("You are pretty healthy.")
elif BMI <= 29.9:
    print("You are slightly overweight.")
elif BMI <= 34.9:
    print("You are severely overweight.")
elif BMI <= 39.9:
    print("You are obese.. -_-")
else:
    print("You are severely obese... Get Some Help")
But what if we want to calculate different health indexes using the same or different input data (like Body Adiposity Index AKA BAI, Waist-to-Hip, etc), this approach has to be failed.
So as previous diagrams explain, we teach computers that this is BMI and how to calculate BMI, this is BAI how to calculate that using given example data and outputs. So those will become Answers and Data according to our previous diagrams. what is happening inside is the computer is learning the pattern of data and answers using machine learning algorithms. So this has opened a new horizon in the computer science and data science world.
Machine Learning Road Map
Introduction to Machine Learning
Supervised Learning and Linear Regression
Classification and Logistic Regression
Decision Tree and Random Forest
Naïve Bayes and Support Vector Machine
Unsupervised Learning
Natural Language Processing and Text Mining
Introduction to Deep Learning
Time Series Analysis
if you want to study machine learning in-depth, I recommend these courses for a successful career through machine learning.
Simple Number Pattern Prediction using Neural Networks 
Here I am going to show you how to do a simple pattern recognition like above using Neural Networks which is a part of Deep Learning.
Deep Learning is a more complicated advanced topic of machine learning but it is easier with neural networks to do simple pattern recognitions using external Python Libraries like NumPy, Pandas, and Tensorflow.
As previously said, machine learning is the computer learning pattern to do certain things. 
So as an example following number sequence has a pattern. try to recognize this pattern.
x = -1, 0, 1, 2, 3, 4
y = -3, -1, 1, 3, 5, 7
so the relationship is,
y = 2x-1
In our mind, you might figure out that between -1 and 3 there is a difference of 2 and between 0 and -1 there is 1 difference. so there might be multiple of x with 2 and + or - of some value. 
So let's implement this in the Python programming language.
Waaaiiiit !!!! Why Python? Python is simple and consistent, already have massive community support worldwide, have tons of libraries and frameworks for machine learning and data science (NumPy, Pandas, Matplotlib, Seaborn for Data Analysis and Visualization, TensorFlow, Keras, Scikit-learn for Machine learning, OpenCV, Pillow for Computer vision, NLTK, spaCy for Natural language processing)
Let's dive into our first code,
First, we import Keras API (application programming interface) from TensorFlow.
Here keras make it really easy with its set of functions for neural networks. In this line of code, we implemented the simplest possible neurons. "Dense" defines the connected neurons. as there is only one layer of dense here, there is only one layer on neurons in here. And input shape of this layer is also the simplest form with one value. 
import keras

model = keras.Sequential([keras.layers.Dense(units=1,input_shape[1])])
But when it comes to machine learning we need to calculate using Linear algebra calculus and more advanced theories but in Tensorflow and Keras those complicated ones are already implemented.
but if you need to optimize and go deep you definitely want Machine Learning Concepts with Advcanec Scientific mathematical Background.
Then we implement loss function and optimizer. The loss function guesses the relation between x and y and gives the measured result data (good or bad, how much loss or close) to the optimizer to make the next guess. Optimizer decides the accuracy of the guess done by the loss function. Optimizers are methods or approaches that adjust the properties of your neural network, such as weights and learning rate, to decrease losses. The main logic is here to make each guess made by loss function to be better than one before.
model.compile(optimizer="sgd", loss='mean_squared_error')
When gues are making some progress with gues becoming better and better the convergence is being used. To study more about convergence visit this site.
Next, we import the NumPy library to do calculations easily.
import numpy as np
next we implement two flat arrays to x and y values.
xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)
In the next cell we figure out how do these xs value fit in ys values by implementing model.fit method for 500 number of times. epochs = 500 means the training loop will do 500 times its guesses.
# Training the model
model.fit(xs, ys, epochs=500)
And the output looks like this
Epoch 1/500
6/6 [==============================] - 0s 36ms/step - loss: 12.7812
Epoch 2/500
6/6 [==============================] - 0s 524us/step - loss: 10.2833
…….
6/6 [==============================] - 0s 157us/step - loss: 3.9143e-05
Epoch 499/500
6/6 [==============================] - 0s 185us/step - loss: 3.8338e-05
Epoch 500/500
6/6 [==============================] - 0s 255us/step - loss: 3.7551e-05
This is called training the model. After running 500 training loops you can see that loss is getting lower. so next we do prediction of our custon number for X using predict method.
print(model.predict([10.0]))
and the output should be close to 19 but instead it showed
[[18.982122]]
This is happening because of insufficient training data.
if we Train the model with 1000 loops answer will look like this.
[[18.999987]]
So with model training, it becomes really close to the real answer.
You can find the source code for this here.
Simple House Price Prediction using Neural Network Basics and Tensorflow
Here we are going to do a real-life scenario to Build a Neural network using TensorFlow.
Here is the simple formula for the prediction
House of Price with one bedroom = 100k
House of Price with one bedroom = 150k
Accordingly, the price of the houses increases by the number of bedrooms. 
Here we going to build a neural network to predict the prices of houses according to their bedrooms.
here is the two x and y data for this
x = 1, 2, 3, 4, 5, 6, 7
y = 1, 1.5, 2, 2.5, 3, 3.5, 4
And Lets start by importing essential libaries for this
import tensorflow as tf
import numpy as np
from tensorflow import keras
Then we implement whole methods as loss function, optimization, and defining number arrays, defining training loops ( Here 1000 epochs ) inside a basic python function called house_model and return model with empty arguments.
def house_model():
    xs = np.array([1.0,2.0,3.0,4.0,5.0,6.0], dtype=float)
    ys = np.array([1.0,1.5,2.0,2.5,3.0,3.5],dtype=float)
    model = tf.keras.models.Sequential([tf.keras.layers.Dense(units=1,input_shape=[1])])
    model.compile(optimizer = 'sgd', loss = 'mean_squared_error')
    model.fit(xs,ys,epochs=1000)
    return model
Then we assign the number of beds to value to new_y and use predict method and execute the house_model function.
new_y = 10.0
model = house_model()
prediction = model.predict([new_y])[0]
After training the data 1000 times the prediction will print as follows
print(prediction)
[5.5149393]
