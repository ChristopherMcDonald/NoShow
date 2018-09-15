# NeuralNet - uses X and Y to train a NeuralNet
# returns the Confusion Matrix and Accuracy of NN
def NeuralNet(optimizer = 'rmsprop', unit1 = 25, unit2 = 25):
    # from dataImport import getData
    # X, Y = getData(False)

    # Split Data into Testing & Training (20/80)
    # from sklearn.model_selection import train_test_split
    # X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.15, random_state = 0)

    # Balance Training Data
    # indexes = y_train[y_train == 0].sample(frac = 0.75).index
    # y_train = y_train.drop(indexes)
    # X_train = X_train.drop(indexes)

    # Feature Scaling
    # from sklearn.preprocessing import StandardScaler
    # sc = StandardScaler()
    # X_train = sc.fit_transform(X_train)
    # X_test = sc.transform(X_test)

    # Importing the Keras libraries and packages
    import keras
    from keras.models import Sequential
    from keras.layers import Dense
    from keras import optimizers

    # Initialising the ANN
    classifier = Sequential()

    # Adding the input layer and the first hidden layer
    classifier.add(Dense(activation="relu", input_dim=49, units=unit1, kernel_initializer='uniform'))

    # Adding the second hidden layer
    classifier.add(Dense(activation="relu", units=unit2, kernel_initializer='uniform'))

    # Adding the output layer
    classifier.add(Dense(activation="sigmoid", units=1, kernel_initializer='uniform'))

    # Compiling the ANN
    classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])

    # return if using GridSearchCV
    return classifier

    # # Fitting the ANN to the Training set
    # classifier.fit(X_train, y_train, batch_size = batch, epochs = epoch)
    #
    # # Predicting the Test set results
    # y_pred = classifier.predict(X_test)
    # y_pred = (y_pred > 0.5)
    #
    # # Making the Confusion Matrix
    # from sklearn.metrics import confusion_matrix
    # from sklearn.metrics import accuracy_score
    # cm = confusion_matrix(y_test, y_pred)
    # acc = accuracy_score(y_test, y_pred)
    #
    # # For Testing and Tuning
    # print("Acc w/ {} {}: {}".format(batch, epoch, acc))
    #
    # return cm, acc
