import sys
sys.path.append('./../Data/')
from KPCA import getComps
import numpy as np

X, Y = getComps(48, True)

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.20, random_state = 0)

# Fitting Kernel SVM to the Training set
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
classifier = gnb.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
cm = confusion_matrix(y_test, y_pred)
acc = accuracy_score(y_test, y_pred)
print(cm)
print("ACC: " + str(acc))
