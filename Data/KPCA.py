import sys
sys.path.append('./../Data/')
from dataImport import getData
import numpy as np
from sklearn.decomposition import PCA

## getComps - pulls the n-best components of the data which account for the most variance
## n - how many components to pull
## balance - Boolean to instruct whether to randomly balance data
def getComps(n, balance):
    # pull data as per usual
    X, Y = getData(balance)

    # use SciKit PCA library
    pca = PCA(n_components=n)
    pca.fit(X)
    X = pca.transform(X)

    return X, Y
