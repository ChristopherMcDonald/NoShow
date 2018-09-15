from dataImport import getData
import numpy
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
import NeuralNet

# NeuralNet.NeuralNet()

# from dataImport import getData
X, Y = getData(False)

# Split Data into Testing & Training (20/80)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.20, random_state = 0)
indexes = y_train[y_train == 0].sample(frac = 0.75).index
y_train = y_train.drop(indexes)
X_train = X_train.drop(indexes)

optimizers = ['adam', 'rmsprop', 'sgd', 'adadelta', 'adagrad', 'adamax', 'nadam'];
batch_size = [8, 16, 32, 64, 128, 256]
epochs = [10, 50, 100]
learn_rate = [0.001, 0.01, 0.1, 0.2, 0.3]
momentum = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9]
param_grid = dict(batch_size = batch_size, epochs = epochs, optimizers = optimizers, learn_rate = learn_rate, momentum = momentum)

# import relevant libraries, use GridSearchCV to find optimal values
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier
model = KerasClassifier(build_fn=NeuralNet.NeuralNet(), verbose=0)
clf = GridSearchCV(model, param_grid, scoring = 'accuracy', n_jobs = -1)
clf.fit(X_train, y_train)

print("Best parameters set found on development set:")
print()
print(clf.best_params_)
print("Grid scores on development set:")
print()
means = clf.cv_results_['mean_test_score']
stds = clf.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, clf.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r"
          % (mean, std * 2, params))
print()

# Algo, LR, Momentum, Layers + Nodes, Dropout, Batch
