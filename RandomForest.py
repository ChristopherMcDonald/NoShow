from KPCA import getComps

X, Y = getComps(3, False)

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.20, random_state = 0)

# crits = ['entropy']
# maxes = ['auto']
# trees = [1024]
#
# params = {'criterion': crits, 'max_features': maxes, 'n_estimators': trees};
#
# # import relevant libraries, use GridSearchCV to find optimal values
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import GridSearchCV
# clf = GridSearchCV(RandomForestClassifier(class_weight = {1: 4, 0: 1}, random_state = 0), params, scoring = 'accuracy', n_jobs = -1)
# clf.fit(X_train, y_train)
#
# print("Best parameters set found on development set:")
# print()
# print(clf.best_params_)
# print("Grid scores on development set:")
# print()
# means = clf.cv_results_['mean_test_score']
# stds = clf.cv_results_['std_test_score']
# for mean, std, params in zip(means, stds, clf.cv_results_['params']):
#     print("%0.3f (+/-%0.03f) for %r"
#           % (mean, std * 2, params))
# print()

# Fitting Kernel SVM to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 1024, criterion = 'entropy', random_state = 0, class_weight={1: 4, 0: 1}, max_features = 'auto')
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
cm = confusion_matrix(y_test, y_pred)
acc = accuracy_score(y_test, y_pred)
