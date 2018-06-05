from KPCA import getComps

X, Y = getComps(3, False)

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.20, random_state = 0)

crits = ['gini', 'entropy']
maxes = ['log2', 'auto', 0.2, 0.4, 0.6, 0.8, 1.0]
trees = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]

params = {'criterion': crits, 'max_features': maxes, 'n_estimators': trees};

# import relevant libraries, use GridSearchCV to find optimal values
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
clf = GridSearchCV(RandomForestClassifier(class_weight = {1: 4, 0: 1}, random_state = 0), params, scoring = 'accuracy', n_jobs = -1)
clf.fit(X_train, y_train)

out = open("output.txt", "w");

out.write("Best parameters set found on development set:")
out.write()
out.write(clf.best_params_)
out.write("Grid scores on development set:")
out.write()
means = clf.cv_results_['mean_test_score']
stds = clf.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, clf.cv_results_['params']):
    out.write("%0.3f (+/-%0.03f) for %r"
          % (mean, std * 2, params))
out.write()
out.close();

# find best params from above and plug into below

# Fitting Kernel SVM to the Training set
# from sklearn.ensemble import RandomForestClassifier
# classifier = RandomForestClassifier(n_estimators = 5, criterion = 'gini', random_state = 0, class_weight={1: 4, 0: 1}, max_features = val2)
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
