from dataImport import getData;
import numpy
from keras.wrappers.scikit_learn import KerasClassifier;
from sklearn.model_selection import GridSearchCV
import NeuralNet;

NeuralNet.NeuralNet();
# algo, LR, Momentum, Layers + Nodes, Dropout, Batch
# Algo:
# RMS prop - 0.5906993576404596
# 
# LR:
# 0.001 - 0.5780458383594692
