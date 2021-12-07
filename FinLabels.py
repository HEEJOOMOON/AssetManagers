import statsmodels.api as sml
import numpy as np
import pandas as pd
from mlfinlab import *
import matplotlib.pyplot as plt

def tValLinR(close):
    # tValue from a linear trend
    x = np.ones((close.shape[0], 2))
    x[:, 1] = np.arange(close.shape[0])
    ols = sml.OLS(close, x).fit()
    return ols.tvalues[1]

def getBinsFromTrend(molecule, close, span):
    ## mlfinlab
    """
    Derive lables from the sign of t-value of lienar trend
    molecule: index of labels
    Output includes:
    - t1: End time for the identified trend
    - tVal: t-value associated with the estimated trend coefficient
    - bin: Sign of the trend
    """
    out = pd.DataFrame(index=molecule, columns=['t1', 'tVal', 'bin'])
    hrzns = range(*span)    # span list로 여러개 받는 경우
    for dt0 in molecule:    # label할 datetime들
        df0 = pd.Series()   # "df0"
        iloc0 = close.index.get_loc(dt0)    # datetime을 숫자 인덱스로 바꿈
        if iloc0+max(hrzns) > close.shape[0]: continue
        for hrzn in hrzns:      # span 여러개인 경우
            dt1 = close.index[iloc0+hrzn-1]     # dt1: t+hrzn datetime
            df1 = close.loc[dt0:dt1]    # label할 첫번째 starting point(dt0) 부터 ending point까지(dt1)
            df0.loc[dt1] = tValLinR(df1.values)     # df0에 ending point index만들어서 t-value값 넣기
        dt1 = df0.replace([-np.inf, np.inf, np.nan], 0).abs().idxmax()  # t-value 절대값이 가장 큰 index
        out.loc[dt0, ['t1', 'tVal', 'bin']] = df0.index[-1], df0[dt1], np.sign(df0[dt1])    # prevent leakage
                                                                                            # df0.index[-1] 가장 긴 L을 Trend가 끝나는 기간으로 봄.
    out['t1'] = pd.to_datetime(out['t1'])
    out['bin'] = pd.to_numeric(out['bin'], downcast='signed')
    return out.dropna(subset=['bin'])

if __name__=='__main__':
    # Testing the trend-scanning labeling algorithm
    df0 = pd.Series(np.random.normal(0, .1, 100)).cumsum()
    df0 += np.sin(np.linspace(0, 10, df0.shape[0]))
    df1 = getBinsFromTrend(df0.index, df0, [3, 10, 1])
    plt.scatter(df1.index, df0.loc[df1.inex].values, c=df1['bin'].values, cmap='viridis')
    plt.show()