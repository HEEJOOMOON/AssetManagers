import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.neighbors import KernelDensity
from utils import *
from metrics import empirical_info

# © 2020 Machine Learning for Asset Managers, Marcos Lopez de Prado

def mpPDF(var, q, pts):
    # Marcenko-Pastur pdf(theoretical)
    # q = T/N
    eMin, eMax = var*(1-(1./q)**.5)**2, var*(1+(1./q)**.5)**2
    eVal = np.linspace(eMin, eMax, pts)
    pdf = q/(2*np.pi*var*eVal)**((eMax-eVal)*(eVal-eMin))**.5
    pdf = pd.Series(pdf.flatten(), index=eVal.flatten())
    return pdf

def getPCA(matrix):
    # Get eVal, eVec from a Hermitian matrix
    eVal, eVec = np.linalg.eigh(matrix)
    indices = eVal.argsort()[::-1]
    eVal, eVec = eVal[indices], eVec[:, indices]
    eVal = np.diagflat(eVal)
    return eVal, eVec, indices

def fitKDE(obs, bWidth=.25, kernel='gaussian', x=None):
    # Fit kernel to a series of obs, and derive the prob of obs
    # empirical pdf
    # x is the array of values on which the fit KDE will be evaluated
    if len(obs.shape)==1: obs=obs.reshape(-1,1)
    kde = KernelDensity(kernel=kernel, bandwidth=bWidth).fit(obs)
    if x is None: x=np.unique(obs).reshape(-1,1)
    if len(x.shape)==1: x=x.reshape(-1,1)
    logProb = kde.score_samples(x)  # log(density)
    pdf = pd.Series(np.exp(logProb), index=x.flatten())
    return pdf

def errPDFs(var, eVal, q, bWidth, pts=100):
    # Fit error
    pdf0 = mpPDF(var, q, pts)   # theoretical pdf
    pdf1 = fitKDE(eVal, bWidth, x=pdf0.index.values)
    sse = np.sum((pdf1-pdf0)**2)
    return sse

def findMaxEval(eVal, q, bWidth):
    # Find max random eVal by fitting Marcenko's dist
    out = minimize(lambda *x:errPDFs(*x), .5, args=(eVal, q, bWidth), bounds=((1E-5, 1-1E-5),))
    if out['success']: var=out['x'][0]
    else: var=1
    eMax = var*(1+(1./q)**.5)**2
    return eMax, var

def denoiseCorr(eVal, eVec, nFacts):
    # Remove noise from corr by fixing random eigenvalues
    eVal_ = np.diag(eVal).copy()
    eVal_[nFacts:] = eVal_[nFacts:].sum() / float(eVal_.shape[0]-nFacts)
    eVal_ = np.diag(eVal_)
    corr1 = np.dot(eVec, eVal_).dot(eVec.T)
    corr1 = cov2corr(corr1)
    return corr1

def denoiseCorr2(eVal, eVec, nFacts, alpha=0):
    # Remove noise from corr through targeted shrinkage
    eValL, eVecL = eVal[:nFacts, :nFacts], eVec[:, :nFacts]
    eValR, eVecR = eVal[nFacts:, nFacts:], eVec[:, nFacts:]
    corr0 = np.dot(eVecL, eValL).dot(eVecL.T)
    corr1 = np.dot(eVecR, eValR).dot(eVecR.T)
    corr2 = corr0 + alpha*corr1 + (1-alpha)*np.diag(np.diag(corr1))
    return corr2

def deNoiseCov(cov0, q, bWidth):
    corr0 = cov2corr(cov0)
    eVal0, eVec0, ind = getPCA(corr0)
    eMax0, var0 = findMaxEval(np.diag(eVal0), q, bWidth)
    nFacts0 = eVal0.shape[0] - np.diag(eVal0)[::-1].searchsorted(eMax0)
    corr1 = denoiseCorr(eVal0, eVec0, nFacts0)
    cov1 = corr2cov(corr1, np.diag(cov0)**.5)
    return cov1

def optPort(cov, mu=None):
    inv = np.linalg.inv(cov)
    ones = np.ones(shape=(inv.shape[0],1))
    if mu is None: mu=ones
    w = np.dot(inv, mu)
    w /= np.dot(ones.T, w)
    return w

def empirical_denoising(returns: pd.DataFrame,
                        q: float,
                        bWidth: float=0.01,
                        alpha: float=.5,
                        metrics: str='corr',
                        ) -> pd.DataFrame:
    '''

    :param returns: (pd.DataFrame) returns of stocks
    :param bWidth: (float) kernel density bandwidth
    :param metrics: (str) types of metrics
        - corr: correlation matrix
        - mutual: mutual information
        - variation: variational information
    :return: (pd.DataFrame) denoised distance metric
    '''

    if metrics=='corr':
        metrics = returns.corr()
    else:
        metrics = empirical_info(returns, norm=False, metrics=metrics)
    eVal0, eVec0, ind = getPCA(metrics)
    eMax0, var0 = findMaxEval(np.diag(eVal0), q, bWidth=bWidth)
    nFacts0 = eVal0.shape[0] - np.diag(eVal0)[::-1].searchsorted(eMax0)
    corr2 = denoiseCorr2(eVal0, eVec0, nFacts0, alpha=alpha)
    idx, cols = returns.columns[ind], returns.columns[ind]
    final = pd.DataFrame(corr2, index=idx, columns=cols)

    return final

if __name__ == '__main__':

    alpha, nCols, nFact, q = .995, 1000, 100, 10
    cov = np.cov(np.random.normal(size=(nCols*q, nCols)), rowvar=0)
    cov = alpha*cov + (1-alpha)*getRndCov(nCols, nFact)     # noise + signal
    corr0 = cov2corr(cov)
    eVal0, eVec0, _ = getPCA(corr0)
    eMax0, var0 = findMaxEval(np.diag(eVal0), q, bWidth=.01)
    nFacts0 = eVal0.shape[0] - np.diag(eVal0)[::-1].searchsorted(eMax0)
    corr1 = denoiseCorr(eVal0, eVec0, nFacts0)
    eVal1, eVec1, _ = getPCA(corr1)
    corr2 = denoiseCorr2(eVal0, eVec0, nFacts0, alpha=.5)
    eVal2, eVec2, _ = getPCA(corr2)