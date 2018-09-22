import sys
sys.path.append('./../Data/')
from dataImport import getData
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

## getComps - pulls the n-best components of the data which account for the most variance
## n - how many components to pull, 28 is good
## balance - Boolean to instruct whether to randomly balance data
def getComps(n, balance):
    # pull data as per usual
    X, Y = getData(True)

    scaler = StandardScaler()
    scaler.fit(X)
    X = scaler.transform(X)

    # use SciKit PCA library
    pca = PCA(n_components=28)
    pca.fit(X)
    X = pca.transform(X)

    return X, Y
