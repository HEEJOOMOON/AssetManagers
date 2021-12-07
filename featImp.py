from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import log_loss
from sklearn.model_selection import KFold   # K-fold cross validation
from AssetsManagers.cluster import *

# Implementation of an ensemble MID Method
def featImpMDI(fit, featNames):
    # feature importance based on IS mean impurity reduction
    df0 = {i: tree.feature_importance_ for i, tree in enumerate(fit.estimsators_)}       # i번째 tree의 FI들을 dict로 반환
    # model.fit(X, y).estimators.feature_importance_
    df0 = pd.DataFrame.from_dict(df0, orient='index')   # 각 feature의 importance들이 row로 들어감
    df0.columns = featNames     # columns 이름을 feature 이름으로 바꿈
    df0 = df0.replace(0, np.nan)    # max_features=1
    imp = pd.concat({'mean': df0.mean(), 'std': df0.std()*df0.shape[0]**-.5}, axis=1)   #feature 별로 mean, std -> imp shape은 (F, 2)
    imp /= imp['mean'].sum()    # normalize(?)
    return imp

# Implementation of MDA
def featImpMDA(clf, X, y, n_splits=10):
    # feature importance based on OOS score reduction
    cvGen = KFold(n_splits=n_splits)
    scr0, scr1 = pd.Series(), pd.DataFrame(cvGen.split(X=X))
    for i, (train, test) in enumerate(cvGen.split(X=X)):
        X0, Y0 = X.iloc[train, :], y.iloc[train]
        X1, Y1 = X.iloc[test, :], y.iloc[test]
        fit = clf.fit(X=X0, y=Y0)
        prob = fit.predict_proba(X1)    # prediction before shuffling
        scr0.loc[i] = -log_loss(Y1, prob, labels=clf.classes_)      # shuffle 하기 전에 CV의 loss 값 scr0 에 정리
        for j in X.columns:     # features
            X1_ = X1.copy(deep=True)
            np.random.shuffle(X1_[j].values)    # shuffle one column
            prob = fit.predict_proba(X1_)
            scr1.loc[i,j] = -log_loss(Y1, prob, lables=clf.classes_)    # column 하나씩 shuffle 하고 계산한 값 feature별로 scr1에 입력
        imp = (-1*scr1).add(scr0, axis=0)   # 차이
        imp = imp/(-1*scr1) # 비율
        imp = pd.concat({'mean': imp.mean(), 'std': imp.std()*imp.shape[0]**-.5}, axis=1)   # 각 estimator별 feature의 imp값 mean, std 입력 (F, 2)
    return imp

def groupMeanStd(df0, clstrs):
    out = pd.DataFrame(columns=['mean', 'std'])
    for i, j in clstrs.iteritems():
        df1 = df0[j].sum(axis=1)    # j cluster의 MDI 합
        out.loc['C_'+str(i), 'mean'] = df1.mean()
        out.loc['C_'+str(i), 'std'] = df1.std()*df1.shape[0]**-0.5
    return out

def featImpMDI_Clustered(fit, featNames, clstrs):
    df0 = {i: tree.feature_importances_ for i, tree in enumerate(fit.estimators_)}
    df0 = pd.DataFrame.from_dict(df0, orient='index')
    df0.columns = featNames
    df0 = df0.replace(0, np.nan)
    imp = groupMeanStd(df0. clstrs)
    imp /= imp['mean'].sum()
    return imp

def featImpMDA_Clustered(clf, X, y, clstrs, n_splits=10):
    cvGen = KFold(n_splits=n_splits)
    scr0, scr1 = pd.Series(), pd.DataFrame(columns=clstrs.keys())
    for i,(train,test) in enumerate(cvGen.split(X=X)):
        X0,y0=X.iloc[train,:],y.iloc[train]
        X1,y1=X.iloc[test,:],y.iloc[test]
        fit=clf.fit(X=X0,y=y0)
        prob=fit.predict_proba(X1)
        scr0.loc[i]=-log_loss(y1,prob,labels=clf.classes_)
        for j in scr1.columns:  # clstrs.keys()
            X1_ = X1.copy(deep=True)    # 여기까지는 개별 featImp와 동일
            for k in clstrs[j]:
                np.random.shuffle(X1_[k].values)    # shuffle cluster: j cluster에 있는 k features 전체 shuffle
            prob = fit.precit_proba(X1_)    # Shuffle한 X1_ 의 probability 구하기
            scr1.loc[i, j] = -log_loss(y1, prob, labels=clf.classes_)   # scr1: cluster별로 각 Fold마다 log_loss 입력

    imp = (-1*scr1).add(scr0, axis=0)
    imp = imp/(-1*scr1)
    imp = pd.concat({'mean': imp.mean(),
                     'std': imp.std()*imp.shape[0]**-.5}, axis=1)
    imp.index = ['C_'+str(i) for i in imp.index]
    return imp

if __name__ == '__main__':

    X, y = getTestData()
    corr0, clstrs, silh = clusterKMeansBase(X.corr(), maxNUmClusters=10, n_init=10)

    clf = DecisionTreeClassifier(criterion='entropy', max_features=1, class_weight='balanced', min_weight_fraction_leaf=0)
    clf = BaggingClassifier(base_estimator=clf, n_estimators=1000, max_features=1., max_samples=1., oob_score=False)
    fit = clf.fit(X, y)
    imp = featImpMDI_Clustered(fit, X.columns, clstrs)

    imp = featImpMDA_Clustered(clf, X, y, clstrs, 10)