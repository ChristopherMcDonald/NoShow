from KPCA import getComps

X, Y = getComps(9, False);

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.25, random_state = 0)

# # Feature Scaling
# from sklearn.preprocessing import StandardScaler
# sc = StandardScaler()
# X_train = sc.fit_transform(X_train)
# X_test = sc.transform(X_test)

vals = ['gini', 'entropy'];
vals2 = ['log2', 'auto'];
vals3 = [5, 10, 15, 20, 25, 30];

for val in vals:
    for val2 in vals2:
        for val3 in vals3:

            # Fitting Kernel SVM to the Training set
            from sklearn.ensemble import RandomForestClassifier
            classifier = RandomForestClassifier(n_estimators = val3, criterion = val, random_state = 0, class_weight={1: 4}, max_features = val2)
            classifier.fit(X_train, y_train)

            # Predicting the Test set results
            y_pred = classifier.predict(X_test)

            # Making the Confusion Matrix
            from sklearn.metrics import confusion_matrix
            from sklearn.metrics import accuracy_score
            cm = confusion_matrix(y_test, y_pred)
            acc = accuracy_score(y_test, y_pred)
            print(cm)
            print("ACC w/ " + val + "," + str(val3) + "," + str(val2) + " : " + str(acc))
