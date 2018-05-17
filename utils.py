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
    return np.power(np.prod(1+data,axis=0),1/days)-1

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

def Volatility(arr,yearly=False):
    cols,days = _check(arr)
    data = np.array(arr).reshape(days, cols)
    if yearly:
        return np.sqrt(np.var(data,axis=0))
    else:
        return np.sqrt((252/days)*np.sum((data-np.mean(data,axis=0))**2,axis=0))

def SharpRatio(arr,rf,yearly = False):
    cols,days = _check(arr)
    c,row = _check(rf)
    if not days == row:
        raise RuntimeError("length of columns of inputs do not match (%s, %s)."% (days,row))
    data = np.array(arr).reshape(days, cols)
    # if not yearly:
    #     data = np.power(1+data,250)-1
    r = np.array(rf).reshape(days,1)*250
    ER = np.power(np.product(1+data,axis=0),250/days)-np.mean(r,axis=0)-1
    #ER = np.mean(data,axis=0) - np.mean(r, axis=0)
    return ER/Volatility(data)

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

def Summary(arr,RF, q=0.99):
    result = arr
    cols,days = _check(result)
    print("Last PnL after %s: " % days,PnL(result)[-1,:])
    # Geometric mean
    print("Geometric mean", gmean(result))
    # min
    print("Daily min", np.min(result, axis=0))
    # max drawdown
    print('max drawdown: ', MaxDrawdown(result))
    # Vol
    print("Volatility", Volatility(result))

    # Sharp Ratio

    print("Sharp ratio: ", SharpRatio(result, RF))
    print("Mean sharp: ", np.mean(SharpRatio(result, RF), axis=0))

    # Kurtosis
    print("Kurtosis: ", Kurtosis(result))
    print("Skewness: ", Skewness(result))
    print("%s VaR %s days: " % (q,days), VaR(result,q))
    print("%s CVaR %s days: " % (q, days), CVaR(result, q))