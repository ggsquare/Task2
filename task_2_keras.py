import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.multiclass import OneVsOneClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils

trainfilename = '/Users/Anna/polybox/IntroductionML/Tasks/02/task2_s82hdj/train.csv'
testfilename = '//Users/Anna/polybox/IntroductionML/Tasks/02/task2_s82hdj/test.csv'

trainfile = pd.read_csv(trainfilename, delimiter = ',')
testfile = pd.read_csv(testfilename, delimiter = ',')

X_test = testfile._drop_axis(['Id'], axis=1)
X_train = trainfile._drop_axis(['Id', 'y'], axis=1)
y_train = trainfile['y']

#OneVsOneClassifier with Linear SVC (Linear Support Vector Classification)
#classifier = OneVsOneClassifier(LinearSVC(random_state=0))
#classifier = SVC()

#OneVsRestClassifier(estimator, n_jobs)
#classifier = OneVsRestClassifier(SVC(kernel='linear'))
#classifier = OneVsRestClassifier(SVC(kernel='rbf'))
#classifier = LinearSVC() #takes rbf as default

#hidden_layer_sizes=(5, 2): 2 hidden layers with 5 and 2 Neurons
#classifier = MLPClassifier(solver='sgd', alpha=1e-5, hidden_layer_sizes=(16, 16, 16, 16, 16, 16,), random_state=1, activation='relu', verbose=True, learning_rate='adaptive')

# create model
classifier = Sequential()
classifier.add(Dense(16, input_dim=16, activation='relu'))
classifier.add(Dense(16, activation='relu'))
classifier.add(Dense(16, activation='relu'))
classifier.add(Dense(3, activation='softmax'))

classifier.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

y_train_encoded = np_utils.to_categorical(y_train.values)

classifier.fit(X_train.values, y_train_encoded, epochs=200)
y_pred = classifier.predict(X_test)

y_pred_class = []
for y in y_pred:
    y_pred_class.append(np.argmax(y))
#acc = accuracy_score(y_train, y_pred)

# output results
d={'Id': testfile['Id'], 'y': y_pred_class}
output=pd.DataFrame(d)
output.to_csv('task_2_output_keras.csv', index=False)
