import sys
sys.path.append('./../Data/')
from KPCA import getComps
import numpy as np

X, Y = getComps(5, True)

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.20, random_state = 0)

# This code can find the most optimal parameters!
# covers a wide range of combinations of parameters for the SVM classifier
# like: kernel, C, degree, gamma, coef (to be used with poly and sigmoid)
params = [
    { 'kernel':['rbf', 'sigmoid', 'poly'], 'gamma':[.001, .01, .1, 1, 10, 100, 100], 'C':[.001, .01, .1, 1, 10, 100, 100], 'shrinking': [True, False]}
]

print("Starting GridSearchCV!")
# import relevant libraries, use GridSearchCV to find optimal values
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import BaggingClassifier
clf = GridSearchCV(SVC(random_state = 0), params, scoring = 'accuracy', n_jobs = 10)
clf.fit(X_train, y_train)

print("Best parameters set found on development set:")
print()
print(clf.best_params_)
print("Grid scores on development set:")
print()
means = clf.cv_results_['mean_test_score']
stds = clf.cv_results_['std_test_score']
f = open("SVM-Final.txt", "w")
for mean, std, params in zip(means, stds, clf.cv_results_['params']):
    f.write("%0.3f (+/-%0.03f) for %r \n" % (mean, std * 2, params))
f.close()
