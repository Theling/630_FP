import numpy as np
import pandas as pd


def PnL(arr):
    if len(np.array(arr).shape)==1:
        days = len(np.array(arr))
        cols = 1
        data = np.array(arr).reshape(days,1)
    elif len(np.array(arr).shape)==2:
        days = np.array(arr).shape[0]
        cols = np.array(arr).shape[1]
        data = np.array(arr).reshape(days,cols)
    else:
        raise TypeError("Input should be 1-D np.array or pd.Series, or 2-D np.array or pd.DataFrame")
    ret = []
    s = np.array([0.0 for _ in range(cols)])
    for day in range(days):
        s += data[day,:]
        ret.append(s)
    return ret