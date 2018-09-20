import sys
sys.path.append('./../Data/')
from KPCA import getComps

X, Y = getComps(7, False)

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.20, random_state = 0)

crits = ['entropy', 'gini']
maxes = ['auto', 0.2, 0.4, 0.6, 0.8, 1.0]
trees = [128, 256, 512, 1024, 2048]
bootstrap = [True, False]

params = {'criterion': crits, 'max_features': maxes, 'n_estimators': trees, 'bootstrap': bootstrap};

# import relevant libraries, use GridSearchCV to find optimal values
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
clf = GridSearchCV(RandomForestClassifier(random_state = 0), params, scoring = 'accuracy', n_jobs = 10)
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
