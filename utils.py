import numpy as np
import pandas as pd
import scipy.stats as ss

def _check(arr):
    if len(np.array(arr).shape)==1:
        days = len(np.array(arr))
        cols = 1
    elif len(np.array(arr).shape)==2:
        days = np.array(arr).shape[0]
        cols = np.array(arr).shape[1]
    else:
        raise TypeError("Input should be 1-D np.array or pd.Series, or 2-D np.array or pd.DataFrame")
    return cols,days

def PnL(arr,P = 1000000):
    cols,days = _check(arr)
    data = np.array(arr).reshape(days, cols)
    ret = []
    s = (np.array([1.0 for _ in range(cols)]))*P
    for i in range(days):
        s += data[i,:]*s
        ret.append(s.copy())
    return np.array(ret)

def gmean(arr):
    cols,days = _check(arr)
    data = np.array(arr).reshape(days, cols)
    return ss.gmean(1+data,axis=0)-1

def MaxDrawdown(arr, n=10):
    cols,days = _check(arr)
    data = np.array(arr).reshape(days, cols)
    D_ = []
    d_ = []
    for day in range(n,days):
        returns = pd.DataFrame(1+data[(day-n):day,:]).cumprod(axis = 0)
        D = returns.cummax(axis=0)-returns
        d = np.array(D)/(np.array(D+returns))
        D_.append(np.max(np.array(D),axis=0))
        d_.append(np.max(np.array(d),axis = 0))
    #print(np.array(D_).shape)
    return np.max(np.array(D_),axis=0),np.max(np.array(d_),axis=0)

def Volatility(arr):
    cols,days = _check(arr)
    data = np.array(arr).reshape(days, cols)
    return np.sqrt(np.var(data,axis=0))

def SharpRatio(arr,rf):
    cols,days = _check(arr)
    c,row = _check(rf)
    if not days == row:
        raise RuntimeError("length of columns of inputs do not match (%s, %s)."% (days,row))
    r = np.array(rf).reshape(days,1)
    data = np.array(arr).reshape(days,cols)
    ER = data-r
    return ER/Volatility(arr)

def Kurtosis(arr):
    cols,days = _check(arr)
    data = np.array(arr).reshape(days, cols)
    return ss.kurtosis(data,axis=0)

def Skewness(arr):
    cols,days = _check(arr)
    data = np.array(arr).reshape(days, cols)
    return ss.skew(data,axis=0)

def VaR(arr,q):
    cols,days = _check(arr)
    data = np.array(arr).reshape(days, cols)
    tmp = np.sort(data,axis=0)
    n = int(np.around((1-q)*days))
    return -tmp[max(0,n-1),:]

def CVaR(arr,q):
    cols,days = _check(arr)
    data = np.array(arr).reshape(days, cols)
    tmp = np.sort(data,axis=0)
    # print(tmp)
    n = int(np.around((1 - q) * days))
    return np.mean(-tmp[0:max(0, n - 1),:],axis=0)