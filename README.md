# Task 2: 3-Class Classification
This was a group project, but the code in this notebook is solely mine. 

This program uses [keras](https://keras.io/). 

### Description
Loading the data and pre-processing was quite similar to previous tasks as the format of the data has stayed consistent.  The only difference now is that the this is a 3-class classification task with the labels as 0, 1, and 2. We read the data from the train and test csv files into pandas dataframes representing the x training data, y training data, and x testing data. After trying One v. One and One v. Rest classification, we decided to implement a neural network in order to exceed the Hard baseline. Using keras, we defined a model, adding a 2 hidden layers, and a 'softmaxed' output layer. We used ReLU as the activation function for the other layers.  Then, calling the keras compile function, we defined the loss function as categorical cross-entropy and used the Adam optimizer. Subsequently, we converted the y labels to an 'encoded' matrix in order to use it in Keras.  We trained the model for 200 epochs and achieved an accuracy of 91.3%. Despite that, we figured the model would perform a little worse once uploaded, since it may have over-fitted just a little. 
