import numpy as np
from scipy.linalg import block_diag
from sklearn.covariance import LedoitWolf
import pandas as pd
from sklearn.datasets import make_classification

def getRndCov(nCols, nFacts):
    w = np.random.normal(size=(nCols, nFacts))
    cov = np.dot(w, w.T)
    cov += np.diag(np.random.uniform(size=nCols))
    return cov

def cov2corr(cov):
    std = np.sqrt(np.diag(cov))
    corr = cov/np.outer(std, std)   # out[i, j] = a[i] * b[j]
    corr[corr<-1], corr[corr>1] = -1, 1
    return corr

def getTestData(n_features=100, n_informative=25, n_redundant=25, \
                n_samples=10000, random_state=0, sigmaStd=.0):
    # generate a random dataset for a classification problem
    np.random.seed(random_state)
    X, y = make_classification(n_samples=n_samples, n_features=n_features-n_redundant,\
                               n_informative=n_informative, n_redundant=0, shuffle=False,\
                               random_state=random_state)
    cols = {'I_'+str(i) for i in range(n_informative)}
    cols += ['N_'+str(i) for i in range(n_features-n_informative-n_redundant)]  # noise
    X, y = pd.DataFrame(X, columns=cols), pd.Series(y)
    i = np.random.choice(range(n_informative), size=n_redundant)
    for k, j in enumerate(i):
        X['R_'+str(k)] = X['I_'+str(j)] + np.random.normal(size=X.shape[0])*sigmaStd    # redundant variable 만들기?
    return X, y

def corr2cov(corr, std):
    cov = corr*np.outer(std, std)
    return cov

def formBlockMatrix(nBlocks, bSize, bCorr):
    block = np.ones((bSize, bSize)) * bCorr
    block[range(bSize), range(bSize)] = 1
    corr = block_diag(*([block]*nBlocks))
    return corr

def formTrueMatrix(nBlocks, bSize, bCorr):
    corr0 = formBlockMatrix(nBlocks, bSize, bCorr)
    corr0 = pd.DataFrame(corr0)
    cols = corr0.columns.tolist()
    np.random.shuffle(cols)
    corr0 = corr0[cols].loc[cols].copy(deep=True)
    std0 = np.random.uniform(.05, .2, corr0.shape[0])
    cov0 = corr2cov(corr0, std0)
    mu0 = np.random.normal(std0, std0, cov0.shape[0]).reshape(-1,1)
    return mu0, cov0


if __name__=='__main__':
    nBlocks, bSize, bCorr = 10, 50, .5
    np.random.seed(0)
    mu0, cov0 = formTrueMatrix(nBlocks, bSize, bCorr)

if __name__=='__main__':
    X, y = getTestData(40, 5, 30, 10000, sigmaStd=.1)