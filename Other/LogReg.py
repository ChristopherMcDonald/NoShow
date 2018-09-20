import sys
sys.path.append('./../Data/')
from KPCA import getComps
from sklearn import linear_model, datasets

X, Y = getComps(1, False);

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.20, random_state = 0)

## Tested with...
# tols = [0.0001, 0.001, 0.01, 0.1]
# cVals = [1e1, 1e2, 1e3, 1e4, 1e5, 1e6]
#
# # solvs = ['newton-cg', 'lbfgs', 'sag']
# # vals = ['l2']
#
# solvs = ['liblinear']
# vals = ['l1']

logreg = linear_model.LogisticRegression(C=10, tol = 0.1, solver = 'liblinear', penalty = 'l1')

logreg.fit(X, Y)
y_pred = logreg.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
cm = confusion_matrix(y_test, y_pred)
acc = accuracy_score(y_test, y_pred)
# print(cm)
print('Acc: {}'.format(acc)); # 80.181%
