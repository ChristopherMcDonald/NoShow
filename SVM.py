from KPCA import getComps
import numpy as np

X, Y = getComps(3, False)

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.20, random_state = 0)

# covers a wide range of combinations of parameters for the SVM classifier
# like: kernel, C, degree, gamma, coef (to be used with poly and sigmoid)
Cs = [0.001, 0.01, 0.1, 1, 10]
gammas = [0.001, 0.01, 0.1, 1]
param_grid = {'C': Cs, 'gamma' : gammas}

print("starting GridSearchCV")
# import relevant libraries, use GridSearchCV to find optimal values
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
clf = GridSearchCV(SVC(class_weight = {1: 4, 0: 1}, random_state = 0, kernel = 'rbf'), param_grid, scoring = 'accuracy', n_jobs = 12)
clf.fit(X_train, y_train)

print("Best parameters set found on development set:")
print()
print(clf.best_params_)
print("Grid scores on development set:")
print()
means = clf.cv_results_['mean_test_score']
stds = clf.cv_results_['std_test_score']
f = open("SVM.txt", "w")
for mean, std, params in zip(means, stds, clf.cv_results_['params']):
    f.write("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
f.close()

# Use Parameters from above into below code!

# # Fitting Kernel SVM to the Training set
# from sklearn.svm import SVC
# classifier = SVC(kernel = 'rbf', random_state = 0, class_weight={1: 4, 0: 1})
# classifier.fit(X_train, y_train)
#
# # Predicting the Test set results
# y_pred = classifier.predict(X_test)
#
# # Making the Confusion Matrix
# from sklearn.metrics import confusion_matrix
# from sklearn.metrics import accuracy_score
# cm = confusion_matrix(y_test, y_pred)
# acc = accuracy_score(y_test, y_pred)
# print(cm)
# print("ACC: " + str(acc))
