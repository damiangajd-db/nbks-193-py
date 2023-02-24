# Databricks notebook source
# MAGIC %md
# MAGIC ## INTRODUCTION
# MAGIC - It’s a Python based scientific computing package targeted at two sets of audiences:
# MAGIC     - A replacement for NumPy to use the power of GPUs
# MAGIC     - Deep learning research platform that provides maximum flexibility and speed
# MAGIC - pros: 
# MAGIC     - Interactively debugging PyTorch. Many users who have used both frameworks would argue that makes pytorch significantly easier to debug and visualize.
# MAGIC     - Clean support for dynamic graphs
# MAGIC     - Organizational backing from Facebook
# MAGIC     - Blend of high level and low level APIs
# MAGIC - cons:
# MAGIC     - Much less mature than alternatives
# MAGIC     - Limited references / resources outside of the official documentation
# MAGIC - I accept you know neural network basics. If you do not know check my tutorial. Because I will not explain neural network concepts detailed, I only explain how to use pytorch for neural network
# MAGIC - Neural Network tutorial: https://www.kaggle.com/kanncaa1/deep-learning-tutorial-for-beginners 
# MAGIC - The most important parts of this tutorial from matrices to ANN. If you learn these parts very well, implementing remaining parts like CNN or RNN will be very easy. 
# MAGIC <br>
# MAGIC <br>**Content:**
# MAGIC 1. [Basics of Pytorch](#1)
# MAGIC     - Matrices
# MAGIC     - Math
# MAGIC     - Variable
# MAGIC 1. [Linear Regression](#2)
# MAGIC 1. [Logistic Regression](#3)
# MAGIC 1. [Artificial Neural Network (ANN)](#4)
# MAGIC 1. [Concolutional Neural Network (CNN)](#5)
# MAGIC 1. Recurrent Neural Network (RNN)
# MAGIC     - https://www.kaggle.com/kanncaa1/recurrent-neural-network-with-pytorch
# MAGIC 1. Long-Short Term Memory (LSTM)
# MAGIC     - https://www.kaggle.com/kanncaa1/long-short-term-memory-with-pytorch

# COMMAND ----------

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

# COMMAND ----------

# MAGIC %md
# MAGIC <a id="1"></a> <br>
# MAGIC ## Basics of Pytorch
# MAGIC ### Matrices
# MAGIC - In pytorch, matrix(array) is called tensors.
# MAGIC - 3*3 matrix koy. This is 3x3 tensor.
# MAGIC - Lets look at array example with numpy that we already know.
# MAGIC     - We create numpy array with np.numpy() method
# MAGIC     - Type(): type of the array. In this example it is numpy
# MAGIC     - np.shape(): shape of the array. Row x Column

# COMMAND ----------

# import numpy library
import numpy as np

# numpy array
array = [[1,2,3],[4,5,6]]
first_array = np.array(array) # 2x3 array
print("Array Type: {}".format(type(first_array))) # type
print("Array Shape: {}".format(np.shape(first_array))) # shape
print(first_array)

# COMMAND ----------

# MAGIC %md
# MAGIC - We looked at numpy array.
# MAGIC - Now examine how we implement tensor(pytorch array)
# MAGIC - import pytorch library with import torch
# MAGIC - We create tensor with torch.Tensor() method
# MAGIC - type: type of the array. In this example it is tensor
# MAGIC - shape: shape of the array. Row x Column

# COMMAND ----------

# import pytorch library
import torch

# pytorch array
tensor = torch.Tensor(array)
print("Array Type: {}".format(tensor.type)) # type
print("Array Shape: {}".format(tensor.shape)) # shape
print(tensor)

# COMMAND ----------

# MAGIC %md
# MAGIC - Allocation is one of the most used technique in coding. Therefore lets learn how to make it with pytorch.
# MAGIC - In order to learn, compare numpy and tensor
# MAGIC     - np.ones() = torch.ones()
# MAGIC     - np.random.rand() = torch.rand()

# COMMAND ----------

# numpy ones
print("Numpy {}\n".format(np.ones((2,3))))

# pytorch ones
print(torch.ones((2,3)))

# COMMAND ----------

# numpy random
print("Numpy {}\n".format(np.random.rand(2,3)))

# pytorch random
print(torch.rand(2,3))

# COMMAND ----------

# MAGIC %md
# MAGIC - Even if when I use pytorch for neural networks, I feel better if I use numpy. Therefore, usually convert result of neural network that is tensor to numpy array to visualize or examine.
# MAGIC - Lets look at conversion between tensor and numpy arrays.
# MAGIC     - torch.from_numpy(): from numpy to tensor
# MAGIC     - numpy(): from tensor to numpy

# COMMAND ----------

# random numpy array
array = np.random.rand(2,2)
print("{} {}\n".format(type(array),array))

# from numpy to tensor
from_numpy_to_tensor = torch.from_numpy(array)
print("{}\n".format(from_numpy_to_tensor))

# from tensor to numpy
tensor = from_numpy_to_tensor
from_tensor_to_numpy = tensor.numpy()
print("{} {}\n".format(type(from_tensor_to_numpy),from_tensor_to_numpy))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Basic Math with Pytorch
# MAGIC - Resize: view()
# MAGIC - a and b are tensor.
# MAGIC - Addition: torch.add(a,b) = a + b
# MAGIC - Subtraction: a.sub(b) = a - b
# MAGIC - Element wise multiplication: torch.mul(a,b) = a * b 
# MAGIC - Element wise division: torch.div(a,b) = a / b 
# MAGIC - Mean: a.mean()
# MAGIC - Standart Deviation (std): a.std()

# COMMAND ----------

# create tensor 
tensor = torch.ones(3,3)
print("\n",tensor)

# Resize
print("{}{}\n".format(tensor.view(9).shape,tensor.view(9)))

# Addition
print("Addition: {}\n".format(torch.add(tensor,tensor)))

# Subtraction
print("Subtraction: {}\n".format(tensor.sub(tensor)))

# Element wise multiplication
print("Element wise multiplication: {}\n".format(torch.mul(tensor,tensor)))

# Element wise division
print("Element wise division: {}\n".format(torch.div(tensor,tensor)))

# Mean
tensor = torch.Tensor([1,2,3,4,5])
print("Mean: {}".format(tensor.mean()))

# Standart deviation (std)
print("std: {}".format(tensor.std()))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Variables
# MAGIC - It accumulates gradients. 
# MAGIC - We will use pytorch in neural network. And as you know, in neural network we have backpropagation where gradients are calculated. Therefore we need to handle gradients. If you do not know neural network, check my deep learning tutorial first because I will not explain detailed the concepts like optimization, loss function or backpropagation. 
# MAGIC - Deep learning tutorial: https://www.kaggle.com/kanncaa1/deep-learning-tutorial-for-beginners
# MAGIC - Difference between variables and tensor is variable accumulates gradients.
# MAGIC - We can make math operations with variables, too.
# MAGIC - In order to make backward propagation we need variables

# COMMAND ----------

# import variable from pytorch library
from torch.autograd import Variable

# define variable
var = Variable(torch.ones(3), requires_grad = True)
var

# COMMAND ----------

# MAGIC %md
# MAGIC - Assume we have equation y = x^2
# MAGIC - Define x = [2,4] variable
# MAGIC - After calculation we find that y = [4,16] (y = x^2)
# MAGIC - Recap o equation is that o = (1/2)*sum(y) = (1/2)*sum(x^2)
# MAGIC - deriavative of o = x
# MAGIC - Result is equal to x so gradients are [2,4]
# MAGIC - Lets implement

# COMMAND ----------

# lets make basic backward propagation
# we have an equation that is y = x^2
array = [2,4]
tensor = torch.Tensor(array)
x = Variable(tensor, requires_grad = True)
y = x**2
print(" y =  ",y)

# recap o equation o = 1/2*sum(y)
o = (1/2)*sum(y)
print(" o =  ",o)

# backward
o.backward() # calculates gradients

# As I defined, variables accumulates gradients. In this part there is only one variable x.
# Therefore variable x should be have gradients
# Lets look at gradients with x.grad
print("gradients: ",x.grad)

# COMMAND ----------

# MAGIC %md
# MAGIC <a id="2"></a> <br>
# MAGIC ### Linear Regression
# MAGIC - Detailed linear regression tutorial is in my machine learning tutorial in part "Regression". I will not explain it in here detailed.
# MAGIC - Linear Regression tutorial: https://www.kaggle.com/kanncaa1/machine-learning-tutorial-for-beginners
# MAGIC - y = Ax + B.
# MAGIC     - A = slope of curve
# MAGIC     - B = bias (point that intersect y-axis)
# MAGIC - For example, we have car company. If the car price is low, we sell more car. If the car price is high, we sell less car. This is the fact that we know and we have data set about this fact.
# MAGIC - The question is that what will be number of car sell if the car price is 100.

# COMMAND ----------

# As a car company we collect this data from previous selling
# lets define car prices
car_prices_array = [3,4,5,6,7,8,9]
car_price_np = np.array(car_prices_array,dtype=np.float32)
car_price_np = car_price_np.reshape(-1,1)
car_price_tensor = Variable(torch.from_numpy(car_price_np))

# lets define number of car sell
number_of_car_sell_array = [ 7.5, 7, 6.5, 6.0, 5.5, 5.0, 4.5]
number_of_car_sell_np = np.array(number_of_car_sell_array,dtype=np.float32)
number_of_car_sell_np = number_of_car_sell_np.reshape(-1,1)
number_of_car_sell_tensor = Variable(torch.from_numpy(number_of_car_sell_np))

# lets visualize our data
import matplotlib.pyplot as plt
plt.scatter(car_prices_array,number_of_car_sell_array)
plt.xlabel("Car Price $")
plt.ylabel("Number of Car Sell")
plt.title("Car Price$ VS Number of Car Sell")
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC - Now this plot is our collected data
# MAGIC - We have a question that is what will be number of car sell if the car price is 100$
# MAGIC - In order to solve this question we need to use linear regression.
# MAGIC - We need to line fit into this data. Aim is fitting line with minimum error.
# MAGIC - **Steps of Linear Regression**
# MAGIC     1. create LinearRegression class
# MAGIC     1. define model from this LinearRegression class
# MAGIC     1. MSE: Mean squared error
# MAGIC     1. Optimization (SGD:stochastic gradient descent)
# MAGIC     1. Backpropagation
# MAGIC     1. Prediction
# MAGIC - Lets implement it with Pytorch

# COMMAND ----------

# Linear Regression with Pytorch

# libraries
import torch      
from torch.autograd import Variable     
import torch.nn as nn 
import warnings
warnings.filterwarnings("ignore")

# create class
class LinearRegression(nn.Module):
    def __init__(self,input_size,output_size):
        # super function. It inherits from nn.Module and we can access everythink in nn.Module
        super(LinearRegression,self).__init__()
        # Linear function.
        self.linear = nn.Linear(input_dim,output_dim)

    def forward(self,x):
        return self.linear(x)
    
# define model
input_dim = 1
output_dim = 1
model = LinearRegression(input_dim,output_dim) # input and output size are 1

# MSE
mse = nn.MSELoss()

# Optimization (find parameters that minimize error)
learning_rate = 0.02   # how fast we reach best parameters
optimizer = torch.optim.SGD(model.parameters(),lr = learning_rate)

# train model
loss_list = []
iteration_number = 1001
for iteration in range(iteration_number):
        
    # optimization
    optimizer.zero_grad() 
    
    # Forward to get output
    results = model(car_price_tensor)
    
    # Calculate Loss
    loss = mse(results, number_of_car_sell_tensor)
    
    # backward propagation
    loss.backward()
    
    # Updating parameters
    optimizer.step()
    
    # store loss
    loss_list.append(loss.data)
    
    # print loss
    if(iteration % 50 == 0):
        print('epoch {}, loss {}'.format(iteration, loss.data))

plt.plot(range(iteration_number),loss_list)
plt.xlabel("Number of Iterations")
plt.ylabel("Loss")
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC - Number of iteration is 1001.
# MAGIC - Loss is almost zero that you can see from plot or loss in epoch number 1000.
# MAGIC - Now we have a trained model.
# MAGIC - While usign trained model, lets predict car prices.

# COMMAND ----------

# predict our car price 
predicted = model(car_price_tensor).data.numpy()
plt.scatter(car_prices_array,number_of_car_sell_array,label = "original data",color ="red")
plt.scatter(car_prices_array,predicted,label = "predicted data",color ="blue")

# predict if car price is 10$, what will be the number of car sell
#predicted_10 = model(torch.from_numpy(np.array([10]))).data.numpy()
#plt.scatter(10,predicted_10.data,label = "car price 10$",color ="green")
plt.legend()
plt.xlabel("Car Price $")
plt.ylabel("Number of Car Sell")
plt.title("Original vs Predicted values")
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC <a id="3"></a> <br>
# MAGIC ### Logistic Regression
# MAGIC - Linear regression is not good at classification.
# MAGIC - We use logistic regression for classification.
# MAGIC - linear regression + logistic function(softmax) = logistic regression
# MAGIC - Check my deep learning tutorial. There is detailed explanation of logistic regression. 
# MAGIC     - https://www.kaggle.com/kanncaa1/deep-learning-tutorial-for-beginners
# MAGIC - **Steps of Logistic Regression**
# MAGIC     1. Import Libraries
# MAGIC     1. Prepare Dataset
# MAGIC         - We use MNIST dataset.
# MAGIC         - There are 28*28 images and 10 labels from 0 to 9
# MAGIC         - Data is not normalized so we divide each image to 255 that is basic normalization for images.
# MAGIC         - In order to split data, we use train_test_split method from sklearn library
# MAGIC         - Size of train data is 80% and size of test data is 20%.
# MAGIC         - Create feature and target tensors. At the next parts we create variable from these tensors. As you remember we need to define variable for accumulation of gradients.
# MAGIC         - batch_size = batch size means is that for example we have data and it includes 1000 sample. We can train 1000 sample in a same time or we can divide it 10 groups which include 100 sample and train 10 groups in order. Batch size is the group size. For example, I choose batch_size = 100, that means in order to train all data only once we have 336 groups. We train each groups(336) that have batch_size(quota) 100. Finally we train 33600 sample one time.
# MAGIC         - epoch: 1 epoch means training all samples one time.
# MAGIC         - In our example: we have 33600 sample to train and we decide our batch_size is 100. Also we decide epoch is 29(accuracy achieves almost highest value when epoch is 29). Data is trained 29 times. Question is that how many iteration do I need? Lets calculate: 
# MAGIC             - training data 1 times = training 33600 sample (because data includes 33600 sample) 
# MAGIC             - But we split our data 336 groups(group_size = batch_size = 100) our data 
# MAGIC             - Therefore, 1 epoch(training data only once) takes 336 iteration
# MAGIC             - We have 29 epoch, so total iterarion is 9744(that is almost 10000 which I used)
# MAGIC         - TensorDataset(): Data set wrapping tensors. Each sample is retrieved by indexing tensors along the first dimension.
# MAGIC         - DataLoader(): It combines dataset and sample. It also provides multi process iterators over the dataset.
# MAGIC         - Visualize one of the images in dataset
# MAGIC     1. Create Logistic Regression Model
# MAGIC         - Same with linear regression.
# MAGIC         - However as you expect, there should be logistic function in model right?
# MAGIC         - In pytorch, logistic function is in the loss function where we will use at next parts.
# MAGIC     1. Instantiate Model
# MAGIC         - input_dim = 28*28 # size of image px*px
# MAGIC         - output_dim = 10  # labels 0,1,2,3,4,5,6,7,8,9
# MAGIC         - create model
# MAGIC     1. Instantiate Loss 
# MAGIC         - Cross entropy loss
# MAGIC         - It calculates loss that is not surprise :)
# MAGIC         - It also has softmax(logistic function) in it.
# MAGIC     1. Instantiate Optimizer 
# MAGIC         - SGD Optimizer
# MAGIC     1. Traning the Model
# MAGIC     1. Prediction
# MAGIC - As a result, as you can see from plot, while loss decreasing, accuracy(almost 85%) is increasing and our model is learning(training).

# COMMAND ----------

# Import Libraries
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split

# COMMAND ----------

# Prepare Dataset
# load data
train = pd.read_csv(r"../input/train.csv",dtype = np.float32)

# split data into features(pixels) and labels(numbers from 0 to 9)
targets_numpy = train.label.values
features_numpy = train.loc[:,train.columns != "label"].values/255 # normalization

# train test split. Size of train data is 80% and size of test data is 20%. 
features_train, features_test, targets_train, targets_test = train_test_split(features_numpy,
                                                                             targets_numpy,
                                                                             test_size = 0.2,
                                                                             random_state = 42) 

# create feature and targets tensor for train set. As you remember we need variable to accumulate gradients. Therefore first we create tensor, then we will create variable
featuresTrain = torch.from_numpy(features_train)
targetsTrain = torch.from_numpy(targets_train).type(torch.LongTensor) # data type is long

# create feature and targets tensor for test set.
featuresTest = torch.from_numpy(features_test)
targetsTest = torch.from_numpy(targets_test).type(torch.LongTensor) # data type is long

# batch_size, epoch and iteration
batch_size = 100
n_iters = 10000
num_epochs = n_iters / (len(features_train) / batch_size)
num_epochs = int(num_epochs)

# Pytorch train and test sets
train = torch.utils.data.TensorDataset(featuresTrain,targetsTrain)
test = torch.utils.data.TensorDataset(featuresTest,targetsTest)

# data loader
train_loader = DataLoader(train, batch_size = batch_size, shuffle = False)
test_loader = DataLoader(test, batch_size = batch_size, shuffle = False)

# visualize one of the images in data set
plt.imshow(features_numpy[10].reshape(28,28))
plt.axis("off")
plt.title(str(targets_numpy[10]))
plt.savefig('graph.png')
plt.show()

# COMMAND ----------

# Create Logistic Regression Model
class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegressionModel, self).__init__()
        # Linear part
        self.linear = nn.Linear(input_dim, output_dim)
        # There should be logistic function right?
        # However logistic function in pytorch is in loss function
        # So actually we do not forget to put it, it is only at next parts
    
    def forward(self, x):
        out = self.linear(x)
        return out

# Instantiate Model Class
input_dim = 28*28 # size of image px*px
output_dim = 10  # labels 0,1,2,3,4,5,6,7,8,9

# create logistic regression model
model = LogisticRegressionModel(input_dim, output_dim)

# Cross Entropy Loss  
error = nn.CrossEntropyLoss()

# SGD Optimizer 
learning_rate = 0.001
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# COMMAND ----------

# Traning the Model
count = 0
loss_list = []
iteration_list = []
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        
        # Define variables
        train = Variable(images.view(-1, 28*28))
        labels = Variable(labels)
        
        # Clear gradients
        optimizer.zero_grad()
        
        # Forward propagation
        outputs = model(train)
        
        # Calculate softmax and cross entropy loss
        loss = error(outputs, labels)
        
        # Calculate gradients
        loss.backward()
        
        # Update parameters
        optimizer.step()
        
        count += 1
        
        # Prediction
        if count % 50 == 0:
            # Calculate Accuracy         
            correct = 0
            total = 0
            # Predict test dataset
            for images, labels in test_loader: 
                test = Variable(images.view(-1, 28*28))
                
                # Forward propagation
                outputs = model(test)
                
                # Get predictions from the maximum value
                predicted = torch.max(outputs.data, 1)[1]
                
                # Total number of labels
                total += len(labels)
                
                # Total correct predictions
                correct += (predicted == labels).sum()
            
            accuracy = 100 * correct / float(total)
            
            # store loss and iteration
            loss_list.append(loss.data)
            iteration_list.append(count)
        if count % 500 == 0:
            # Print Loss
            print('Iteration: {}  Loss: {}  Accuracy: {}%'.format(count, loss.data, accuracy))

# COMMAND ----------

# visualization
plt.plot(iteration_list,loss_list)
plt.xlabel("Number of iteration")
plt.ylabel("Loss")
plt.title("Logistic Regression: Loss vs Number of iteration")
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC <a id="4"></a> <br>
# MAGIC ### Artificial Neural Network (ANN)
# MAGIC - Logistic regression is good at classification but when complexity(non linearity) increases, the accuracy of model decreases.
# MAGIC - Therefore, we need to increase complexity of model.
# MAGIC - In order to increase complexity of model, we need to add more non linear functions as hidden layer. 
# MAGIC - I am saying again that if you do not know what is artificial neural network check my deep learning tutorial because I will not explain neural network detailed here, only explain pytorch.
# MAGIC - Artificial Neural Network tutorial: https://www.kaggle.com/kanncaa1/deep-learning-tutorial-for-beginners
# MAGIC - What we expect from artificial neural network is that when complexity increases, we use more hidden layers and our model can adapt better. As a result accuracy increase.
# MAGIC - **Steps of ANN:**
# MAGIC     1. Import Libraries
# MAGIC         - In order to show you, I import again but we actually imported them at previous parts.
# MAGIC     1. Prepare Dataset
# MAGIC         - Totally same with previous part(logistic regression).
# MAGIC         - We use same dataset so we only need train_loader and test_loader. 
# MAGIC         - We use same batch size, epoch and iteration numbers.
# MAGIC     1. Create ANN Model
# MAGIC         - We add 3 hidden layers.
# MAGIC         - We use ReLU, Tanh and ELU activation functions for diversity.
# MAGIC     1. Instantiate Model Class
# MAGIC         - input_dim = 28*28 # size of image px*px
# MAGIC         - output_dim = 10  # labels 0,1,2,3,4,5,6,7,8,9
# MAGIC         - Hidden layer dimension is 150. I only choose it as 150 there is no reason. Actually hidden layer dimension is hyperparameter and it should be chosen and tuned. You can try different values for hidden layer dimension and observe the results.
# MAGIC         - create model
# MAGIC     1. Instantiate Loss
# MAGIC         - Cross entropy loss
# MAGIC         - It also has softmax(logistic function) in it.
# MAGIC     1. Instantiate Optimizer
# MAGIC         - SGD Optimizer
# MAGIC     1. Traning the Model
# MAGIC     1. Prediction
# MAGIC - As a result, as you can see from plot, while loss decreasing, accuracy is increasing and our model is learning(training). 
# MAGIC - Thanks to hidden layers model learnt better and accuracy(almost 95%) is better than accuracy of logistic regression model.

# COMMAND ----------

# Import Libraries
import torch
import torch.nn as nn
from torch.autograd import Variable

# COMMAND ----------

# Create ANN Model
class ANNModel(nn.Module):
    
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(ANNModel, self).__init__()
        
        # Linear function 1: 784 --> 150
        self.fc1 = nn.Linear(input_dim, hidden_dim) 
        # Non-linearity 1
        self.relu1 = nn.ReLU()
        
        # Linear function 2: 150 --> 150
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        # Non-linearity 2
        self.tanh2 = nn.Tanh()
        
        # Linear function 3: 150 --> 150
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        # Non-linearity 3
        self.elu3 = nn.ELU()
        
        # Linear function 4 (readout): 150 --> 10
        self.fc4 = nn.Linear(hidden_dim, output_dim)  
    
    def forward(self, x):
        # Linear function 1
        out = self.fc1(x)
        # Non-linearity 1
        out = self.relu1(out)
        
        # Linear function 2
        out = self.fc2(out)
        # Non-linearity 2
        out = self.tanh2(out)
        
        # Linear function 2
        out = self.fc3(out)
        # Non-linearity 2
        out = self.elu3(out)
        
        # Linear function 4 (readout)
        out = self.fc4(out)
        return out

# instantiate ANN
input_dim = 28*28
hidden_dim = 150 #hidden layer dim is one of the hyper parameter and it should be chosen and tuned. For now I only say 150 there is no reason.
output_dim = 10

# Create ANN
model = ANNModel(input_dim, hidden_dim, output_dim)

# Cross Entropy Loss 
error = nn.CrossEntropyLoss()

# SGD Optimizer
learning_rate = 0.02
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# COMMAND ----------

# ANN model training
count = 0
loss_list = []
iteration_list = []
accuracy_list = []
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):

        train = Variable(images.view(-1, 28*28))
        labels = Variable(labels)
        
        # Clear gradients
        optimizer.zero_grad()
        
        # Forward propagation
        outputs = model(train)
        
        # Calculate softmax and ross entropy loss
        loss = error(outputs, labels)
        
        # Calculating gradients
        loss.backward()
        
        # Update parameters
        optimizer.step()
        
        count += 1
        
        if count % 50 == 0:
            # Calculate Accuracy         
            correct = 0
            total = 0
            # Predict test dataset
            for images, labels in test_loader:

                test = Variable(images.view(-1, 28*28))
                
                # Forward propagation
                outputs = model(test)
                
                # Get predictions from the maximum value
                predicted = torch.max(outputs.data, 1)[1]
                
                # Total number of labels
                total += len(labels)

                # Total correct predictions
                correct += (predicted == labels).sum()
            
            accuracy = 100 * correct / float(total)
            
            # store loss and iteration
            loss_list.append(loss.data)
            iteration_list.append(count)
            accuracy_list.append(accuracy)
        if count % 500 == 0:
            # Print Loss
            print('Iteration: {}  Loss: {}  Accuracy: {} %'.format(count, loss.data, accuracy))

# COMMAND ----------

# visualization loss 
plt.plot(iteration_list,loss_list)
plt.xlabel("Number of iteration")
plt.ylabel("Loss")
plt.title("ANN: Loss vs Number of iteration")
plt.show()

# visualization accuracy 
plt.plot(iteration_list,accuracy_list,color = "red")
plt.xlabel("Number of iteration")
plt.ylabel("Accuracy")
plt.title("ANN: Accuracy vs Number of iteration")
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC <a id="5"></a> <br>
# MAGIC ### Convolutional Neural Network (CNN)
# MAGIC - CNN is well adapted to classify images.
# MAGIC - You can learn CNN basics: https://www.kaggle.com/kanncaa1/convolutional-neural-network-cnn-tutorial
# MAGIC - **Steps of CNN:**
# MAGIC     1. Import Libraries
# MAGIC     1. Prepare Dataset
# MAGIC         - Totally same with previous parts.
# MAGIC         - We use same dataset so we only need train_loader and test_loader. 
# MAGIC     1. Convolutional layer: 
# MAGIC         - Create feature maps with filters(kernels).
# MAGIC         - Padding: After applying filter, dimensions of original image decreases. However, we want to preserve as much as information about the original image. We can apply padding to increase dimension of feature map after convolutional layer.
# MAGIC         - We use 2 convolutional layer.
# MAGIC         - Number of feature map is out_channels = 16
# MAGIC         - Filter(kernel) size is 5*5
# MAGIC     1. Pooling layer: 
# MAGIC         - Prepares a condensed feature map from output of convolutional layer(feature map) 
# MAGIC         - 2 pooling layer that we will use max pooling.
# MAGIC         - Pooling size is 2*2
# MAGIC     1. Flattening: Flats the features map
# MAGIC     1. Fully Connected Layer: 
# MAGIC         - Artificial Neural Network that we learnt at previous part.
# MAGIC         - Or it can be only linear like logistic regression but at the end there is always softmax function.
# MAGIC         - We will not use activation function in fully connected layer.
# MAGIC         - You can think that our fully connected layer is logistic regression.
# MAGIC         - We combine convolutional part and logistic regression to create our CNN model.
# MAGIC     1. Instantiate Model Class
# MAGIC         - create model
# MAGIC     1. Instantiate Loss
# MAGIC         - Cross entropy loss
# MAGIC         - It also has softmax(logistic function) in it.
# MAGIC     1. Instantiate Optimizer
# MAGIC         - SGD Optimizer
# MAGIC     1. Traning the Model
# MAGIC     1. Prediction
# MAGIC - As a result, as you can see from plot, while loss decreasing, accuracy is increasing and our model is learning(training). 
# MAGIC - Thanks to convolutional layer, model learnt better and accuracy(almost 98%) is better than accuracy of ANN. Actually while tuning hyperparameters, increase in iteration and expanding convolutional neural network can increase accuracy but it takes too much running time that we do not want at kaggle.

# COMMAND ----------

# Import Libraries
import torch
import torch.nn as nn
from torch.autograd import Variable

# COMMAND ----------

# Create CNN Model
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        
        # Convolution 1
        self.cnn1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=0)
        self.relu1 = nn.ReLU()
        
        # Max pool 1
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
     
        # Convolution 2
        self.cnn2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=0)
        self.relu2 = nn.ReLU()
        
        # Max pool 2
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        
        # Fully connected 1
        self.fc1 = nn.Linear(32 * 4 * 4, 10) 
    
    def forward(self, x):
        # Convolution 1
        out = self.cnn1(x)
        out = self.relu1(out)
        
        # Max pool 1
        out = self.maxpool1(out)
        
        # Convolution 2 
        out = self.cnn2(out)
        out = self.relu2(out)
        
        # Max pool 2 
        out = self.maxpool2(out)
        
        # flatten
        out = out.view(out.size(0), -1)

        # Linear function (readout)
        out = self.fc1(out)
        
        return out

# batch_size, epoch and iteration
batch_size = 100
n_iters = 2500
num_epochs = n_iters / (len(features_train) / batch_size)
num_epochs = int(num_epochs)

# Pytorch train and test sets
train = torch.utils.data.TensorDataset(featuresTrain,targetsTrain)
test = torch.utils.data.TensorDataset(featuresTest,targetsTest)

# data loader
train_loader = torch.utils.data.DataLoader(train, batch_size = batch_size, shuffle = False)
test_loader = torch.utils.data.DataLoader(test, batch_size = batch_size, shuffle = False)
    
# Create CNN
model = CNNModel()

# Cross Entropy Loss 
error = nn.CrossEntropyLoss()

# SGD Optimizer
learning_rate = 0.1
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


# COMMAND ----------

# CNN model training
count = 0
loss_list = []
iteration_list = []
accuracy_list = []
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        
        train = Variable(images.view(100,1,28,28))
        labels = Variable(labels)
        
        # Clear gradients
        optimizer.zero_grad()
        
        # Forward propagation
        outputs = model(train)
        
        # Calculate softmax and ross entropy loss
        loss = error(outputs, labels)
        
        # Calculating gradients
        loss.backward()
        
        # Update parameters
        optimizer.step()
        
        count += 1
        
        if count % 50 == 0:
            # Calculate Accuracy         
            correct = 0
            total = 0
            # Iterate through test dataset
            for images, labels in test_loader:
                
                test = Variable(images.view(100,1,28,28))
                
                # Forward propagation
                outputs = model(test)
                
                # Get predictions from the maximum value
                predicted = torch.max(outputs.data, 1)[1]
                
                # Total number of labels
                total += len(labels)
                
                correct += (predicted == labels).sum()
            
            accuracy = 100 * correct / float(total)
            
            # store loss and iteration
            loss_list.append(loss.data)
            iteration_list.append(count)
            accuracy_list.append(accuracy)
        if count % 500 == 0:
            # Print Loss
            print('Iteration: {}  Loss: {}  Accuracy: {} %'.format(count, loss.data, accuracy))

# COMMAND ----------

# visualization loss 
plt.plot(iteration_list,loss_list)
plt.xlabel("Number of iteration")
plt.ylabel("Loss")
plt.title("CNN: Loss vs Number of iteration")
plt.show()

# visualization accuracy 
plt.plot(iteration_list,accuracy_list,color = "red")
plt.xlabel("Number of iteration")
plt.ylabel("Accuracy")
plt.title("CNN: Accuracy vs Number of iteration")
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Conclusion
# MAGIC In this tutorial, we learn: 
# MAGIC 1. Basics of pytorch
# MAGIC 1. Linear regression with pytorch
# MAGIC 1. Logistic regression with pytorch
# MAGIC 1. Artificial neural network with with pytorch
# MAGIC 1. Convolutional neural network with pytorch
# MAGIC 1. Recurrent neural network with pytorch
# MAGIC     - https://www.kaggle.com/kanncaa1/recurrent-neural-network-with-pytorch
# MAGIC 1. Long-Short Term Memory (LSTM)
# MAGIC     - https://www.kaggle.com/kanncaa1/long-short-term-memory-with-pytorch
# MAGIC 
# MAGIC <br> **If you have any question or suggest, I will be happy to hear it **
