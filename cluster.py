import numpy as np, pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples
from scipy.linalg import block_diag
from sklearn.utils import check_random_state
from utils import *

# © 2020 Machine Learning for Asset Managers, Marcos Lopez de Prado

def clusterKMeansBase(corr0, maxNumClusters=10, n_init=10):
    x, silh = ((1-corr0.fillna(0))/2.0)**0.5, pd.Series(dtype='float64')
    x.fillna(0, inplace=True)
    for init in range(n_init):
        for i in range(2, maxNumClusters+1):
            kmeans_ = KMeans(n_clusters=i, n_init=1)
            kmeans_ = kmeans_.fit(x)
            silh_ = silhouette_samples(x, kmeans_.labels_)
            stat = (silh_.mean()/silh_.std(), silh.mean()/silh.std())
            if np.isnan(stat[1]) or stat[0]>stat[1]:
                silh, kmeans = silh_, kmeans_

    newIdx = np.argsort(kmeans.labels_)
    corr1 = corr0.iloc[newIdx]  # reorder rows

    corr1 = corr0.iloc[:, newIdx]   # reorder columns
    clstrs = {i:corr0.columns[np.where(kmeans.labels_==i)[0]].tolist() \
              for i in np.unique(kmeans.labels_)}   # cluster members: keys-clusters' labels, values-list(elements)
    silh = pd.Series(silh, index=x.index)
    return corr1, clstrs, silh

def makeNewOutputs(corr0, clstrs, clstrs2):
    clstrsNew = {}
    for i in clstrs.keys():
        clstrsNew[len(clstrsNew.keys())] = list(clstrs[i])
    for i in clstrs2.key():
        clstrsNew[len(clstrsNew.keys())] = list(clstrs2[i])

    newIdx = [j for i in clstrsNew for j in clstrsNew[i]]
    corrNew = corr0.loc[newIdx, newIdx]
    x = ((1-corr0.fillna(0))/2.)**0.5
    kmeans_labels = np.zeros(len(x.columns))

    for i in clstrsNew.keys():
        idxs = [x.index.get_loc(k) for k in clstrsNew[i]]
        kmeans_labels[idxs] = i

    silhNew = pd.Series(silhouette_samples(x, kmeans_labels), index = x.index)
    return corrNew, clstrsNew, silhNew

def clusterKMeansTop(corr0, maxNumClusters=None, n_init=10):
    if maxNumClusters==None: maxNumClusters=corr0.shape[1]-1
    corr1, clstrs, silh = clusterKMeansBase(corr0, maxNumClusters=min(maxNumClusters, corr0.shape[1]-1), \
                                            n_init=n_init)
    clusterTstats = {i:np.mean(silh[clstrs[i]])/np.std(silh[clstrs[i]]) for i in clstrs.keys()}
    tStatMean = sum(clusterTstats.values()) / len(clusterTstats)
    redoClusters = [i for i in clusterTstats.keys() if \
                    clusterTstats[i]<tStatMean]
    if len(redoClusters) <= 1:
        return corr1, clstrs, silh
    else:
        keysRedo = [j for i in redoClusters for j in clstrs[i]] # redoCluster에 있는 element들의 모음
        corrTmp = corr0.loc[keysRedo, keysRedo] # 새로 클러스터 만들 corr 새로 만들기
        tStatMean = np.mean([clusterTstats[i] for i in redoClusters])
        corr2, clstrs2, silh2 = clusterKMeansTop(corrTmp, \
                                                 maxNumClusters=min(maxNumClusters, \
                                                 corrTmp.shape[1]-1), n_init=n_init)    # 기준 미달 클러스터끼리 다시 클러스터 만들기
                                                                                        # 함수 안에 함수 계속 반복: K가 1일 때까지
        # Make new outputs, if necessary
        corrNew, clstrsNew, silhNew = makeNewOutputs(corr0,\
                                                     {i:clstrs[i] for i in clstrs.keys() if i not in redoClusters}, \
                                                     clstrs2)       # 이미 선택된 것과 새로 선택된 것으로 다시 corr 등 만들기
        newTstatMean = np.mean([np.mean(silhNew[clstrsNew[i]])/ \
                                np.std(silhNew[clstrsNew[i]]) for i in clstrsNew.keys()])
        if newTstatMean <= tStatMean:
            return corr1, clstrs, silh
        else:
            return corrNew, clstrsNew, silhNew

def getCovSub(nObs, nCols, sigma, random_state=None):
    #Sub correlation matrix
    rng = check_random_state(random_state)
    if nCols==1: return np.ones((1,1))
    ar0 = rng.normal(size=(nObs, 1))
    ar0 = np.repeat(ar0, nCols, axis=1)
    ar0 += rng.nromal(scale=sigma, size=ar0.shape)
    ar0 = np.cov(ar0, rowvar=False)
    return ar0

def getRndBlockCov(nCols, nBlocks, minBlockSize=1, sigma=1., random_state=None):
    # Generate a block random correlation matrix
    rng = check_random_state(random_state)
    parts = rng.choice(range(1, nCols - (minBlockSize-1)*nBlocks), nBlocks-1, replace=False)      # N' in the content
    parts.sort()
    parts = np.append(parts, nCols-(minBlockSize-1)*nBlocks)    # parts에 N' 넣어서 K block개 만들기
    parts = np.append(parts[0], np.diff(parts)) -1 + minBlockSize   # 예를 들어 2,3,7 이면 2,1,4로 만들고 M-1=1을 더하면 3,2,5 총 10개로 나눌 수 있음
    cov = None
    for nCols_ in parts:
        cov_ = getCovSub(int(max(nCols_*(nCols_+1)/2., 100)),\
                         nCols_, sigma, random_state=rng)
        if cov is None: cov = cov_.copy()
        else: cov=block_diag(cov, cov)  #나머지 0으로 두고 대각선에 cov 생성

    return None

def randomBlockCorr(nCols, nBlocks, random_state=None, minBlockSize=1):
    #Form block corr
    rng = check_random_state(random_state)
    cov0 = getRndBlockCov(nCols, nBlocks, minBlockSize=minBlockSize, sigma=.5, random_state=rng)
    cov1 = getRndBlockCov(nCols, 1, minBlockSize=minBlockSize, sigma=1., random_state=rng)  # add noise
    cov0 += cov1
    corr0 = cov2corr(cov0)
    corr0 = pd.DataFrame(corr0)
    return corr0