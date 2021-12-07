from AssetsManagers.cluster import *
from mpPDF import *

# © 2020 Machine Learning for Asset Managers, Marcos Lopez de Prado

def optPort_nco(cov, mu=None, maxNumClusters=None):
    cov = pd.DataFrame(cov)
    if mu is not None: mu=pd.Series(mu[:,0])
    corr1 = cov2corr(cov)
    corr1, clstrs, _ = clusterKMeansBase(corr1, maxNumClusters, n_init=10)
    wIntra = pd.DataFrame(0, index=cov.index, columns=clstrs.keys())    # index: features / columns: 각 cluster
    for i in clstrs:    # e.g. 1~10 까지 각 클러스터 loop
        cov_ = cov.loc[clstrs[i], clstrs[i]].values     # cov에서 clstrs의 i's cluster 해당되는 elements만 선택
        if mu is None: mu_=None
        else: mu_=mu.loc[clstrs[i], clstrs[i]].values.reshape(-1, 1)
        wIntra.loc[clstrs[i], i] = optPort(cov_, mu_).flatten()     # cluster 내에서 optPort한 weights를 wIntra에 입력
                                                                    # 확인해볼 사항: elements에 해당 안되는 (index,i) 에는 nan or 0 값?
    cov_ = wIntra.T.dot(np.dot(cov, wIntra))    # reduce covariance matrix: (clstrsXcov) (covXcov) (covXclstrs) -> (clstrsXclstrs)
    mu_ = (None if mu is None else wIntra.T.dot(mu))
    wInter = pd.Series(optPort(cov_, mu_).flatten(), index=cov_.index)      # cluster간에 optPort 통해 weights 구하기
    nco = wIntra.mul(wInter, axis=1).sum(axis=1).values.reshape(-1, 1)
    # i column의 elements & i clstrs 곱하기 해서 각 요소별 & 각 clustr 별 weights 구했으니까 각 요소별 weight 구하려면 clstrs 합치기
    # sum or mul 등에서 axis 헷갈리는 경우: 1이면 column, 0이면 index -> 합의 결과가 column 생성, row 생성으로 이해
    return nco

if __name__ == '__main__':
    # Correlation cluster step
    alpha, nCols, nFact, q = .995, 1000, 100, 10
    cov0 = np.cov(np.random.normal(size=(nCols * q, nCols)), rowvar=0)
    cov0 = alpha * cov0 + (1 - alpha) * getRndCov(nCols, nFact)  # noise + signal
    cov0 = pd.DataFrame(cov0)
    cols = cov0.columns
    cov1 = deNoiseCov(cov0, q, bWidth=.01)      # denoise cov
    cov1 = pd.DataFrame(cov1, index=cols, columns=cols)
    corr1 = cov2corr(cov1)
    corr, clstrs, silh = clusterKMeansBase(corr1, maxNumClusters=corr0.shape[0]/2, n_init=10)

    # Intracluster Optimal Allocation
    wIntra = pd.DataFrame(0, index=cov1.index, columns=clstrs.keys())
    for i in clstrs:
        wIntra.loc[clstrs[i], i] = optPort(cov1.loc[clstrs[i], clstrs[i]]).flatten()
    cov2 = wIntra.T.dot(np.dot(cov, wIntra))    # reduced covariance matrix

    # Intercluster Optimal Allocations
    wInter = pd.Sereis(optPort(cov2).flatten(), index=cov2.index)
    wAll0 = wIntra.mul(wInter, axis=1).sum(axis=1).sort_index()
