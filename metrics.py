import numpy as np
import scipy.stats as ss
from sklearn.metrics import mutual_info_score
import pandas as pd
from itertools import combinations

# © 2020 Machine Learning for Asset Managers, Marcos Lopez de Prado

def basic_information(x, y, bins, norm=False):
    cXY = np.histogram2d(x, y, bins) [0]
    hX = ss.entropy(np.histogram(x, bins))[0]
    hY = ss.entropy(np.histogram(y, bins))[0]
    iXY = mutual_info_score(None, None, contingency=cXY)
    iXYn = iXY/min(hX, hY)  # normalized mutual information
    hXY = hX+hY - iXY   # joint
    hX_Y = hXY - hY     # conditional
    hY_X = hXY - hX     # conditional
    vXY = hX + hY - 2*iXY   # variation of information
    if norm:
        hXY = hX+hY-iXY     # joint
        vXY /=hXY   # normalized variation of information

    return hX_Y, hY_X, hXY, iXY, vXY

def numBins(nObs, corr=None):
    # Optimal number of bins for discretization
    if corr is None:    # univariate case
        z = (8 + 324*nObs +12*(36*nObs + 729*nObs**2)**0.5)**(1/3.)
        b = round(z/6. + 2./(3*z) + 1/3.)
    else:   # bivariate case
        b = round(2**-.5*(1+(1+24*nObs/(1.-corr**2))**0.5)**0.5)
    return int(b)

def varInfo(x, y, norm=False):
    # Optimal bins
    bXY = numBins(x.shape[0], corr=np.corrcoef(x, y)[0, 1])
    cXY = np.histogram2d(x, y, bXY)[0]
    iXY = mutual_info_score(None, None, contingency=cXY)
    hX = ss.entropy(np.histogram(x, bXY)[0])
    hY = ss.entropy(np.histogram(y, bXY)[0])
    vXY = hX + hY - 2*iXY   # variation of information
    if norm:
        hXY = hX+hY-iXY     # joint
        vXY /=hXY   # normalized variation of information
    return vXY

def mutual_info(x, y, norm=False):
    bXY = numBins(x.shape[0], corr=np.corrcoef(x,y)[0,1])
    cXY = np.histogram2d(x, y, bXY)[0]
    iXY = mutual_info_score(None, None, contingency=cXY)
    if norm:
        hX = ss.entropy(np.histogram(x, bXY)[0])
        hY = ss.entropy(np.histogram(y, bXY)[0])
        iXY/=min(hX, hY)
    return iXY

def empirical_info(returns: pd.DataFrame,
                   norm: bool,
                   metrics: str,
                   ) -> pd.DataFrame:
    '''

    :param returns: (pd.DataFrame) returns of stocks
    :param norm: (bool) normalization
    :param metrics: (str) 'mutual' or 'variation'
    :return: (pd.DataFrame) information matrics
    '''

    info = pd.DataFrame(0, index=returns.columns, columns=returns.columns)

    if metrics=='mutual':
        for i, j in combinations(returns.columns, 2):
            tmp = mutual_info(returns[i], returns[j])
            info.loc[i, j] = tmp
            info.loc[j, i] = tmp
    elif metrics=='variation':
        for i, j in combinations(returns.columns, 2):
            tmp = mutual_info(returns[i], returns[j])
            info.loc[i, j] = tmp
            info.loc[j, i] = tmp
    else:
        raise KeyError('metrics is mutual or variation')
    return info

if __name__=='__main__':

    df = pd.read_csv('data/snp_c1.csv', index_col='Date')
    returns = df.pct_change().dropna()
    q = len(returns.index) / len(returns.columns)
    info = empirical_info(returns, norm=False, metrics='mutual')
