import matplotlib.pyplot as plt
from pandas_datareader.data import DataReader
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.linear_model import LinearRegression
from cvxopt import matrix, solvers
from utils import PnL, gmean, MaxDrawdown, Volatility, SharpRatio, Kurtosis, Skewness, VaR, CVaR, Summary

#download data of ETFs
ticker = ["FXE","EWJ","GLD","QQQ","SPY","SHV","DBA","USO",
          "XBI","ILF","GAF","EPP","FEZ"]
start =  datetime(2007, 1, 1)
end = datetime(2016, 10, 20)
data = pd.DataFrame()

# for i in ticker:
#     data[i] = DataReader(i, 'yahoo', start, end)["Close"]
# data.to_csv("ETFs.csv")

#load data
ETF = pd.read_csv("ETFs.csv", index_col=0)[55:]
F = pd.read_csv("Factors.csv", index_col=0)[56:]
F.index = ETF.index[1:]
#calculte the simples anualized returns for the ETFs
R = (ETF.pct_change(1)[1:])*250
#calculate the excess annualized return for the ETFs
ER = pd.DataFrame(R.values-F["RF"].values.reshape(-1,1),
                  index=F.index, columns=ticker)

F = F.iloc[:,0:3]

# # Before
# R_bc = R["2007-03-23":"2008-03-23"].values
# ER_bc = ER["2007-03-23":"2008-03-23"].values
# F_bc = F["2007-03-23":"2008-03-23"].values

# During
# R_bc = R["2008-03-24":"2009-06-30"].values
# ER_bc = ER["2008-03-24":"2009-06-30"].values
# F_bc = F["2008-03-24":"2009-06-30"].values

# After crisis("2007-03-26"-"2008-03-23")
R_bc = R["2009-06-30":"2016-10-20"].values
ER_bc = ER["2009-06-30":"2016-10-20"].values
F_bc = F["2009-06-30":"2016-10-20"].values

Num_days = len(F_bc)
FR_bc = F_bc[1:].copy()
#short term model(60 days)
Lambda = 0.001
beta_T = [0.5, 1, 1.5]
R_opt = {}

#conduct the max return strategy
window = 63
alocate = 5
R_opt["SPY"] = R_bc[window:,4]/250


for j in beta_T:
    Rp = []
    wp = np.ones((13,1))*1/13
    for i in range(window,Num_days):
        future_return = R_bc[i, :].reshape(-1, 1)
        if i%alocate==0:
            r = R_bc[(i-window):i,:]
            er = ER_bc[(i-window):i,:]
            f1 = F_bc[(i-window):i,:]
            rho = r.mean(axis=0).reshape(-1,1)
            cov_f = np.cov(f1, rowvar=False)
            #run regression to get the beta
            lm = LinearRegression()
            lm.fit(f1, er)
            coeff3 = lm.coef_
            beta = coeff3[:,0]
            error = er - lm.predict(f1)
            #calculate the covariance matrix
            # Q = coeff3.dot(cov_f).dot(coeff3.T)+np.diag(error.var(axis=0))
            Q = np.diag([1]*13)
            #preparation for the optimization
            P = matrix(2*Lambda*Q, tc='d')
            q = matrix(-2*Lambda*(Q.T).dot(wp)-rho, tc='d')
            A = matrix(np.vstack((beta, [1]*13)), tc='d')
            G = matrix(np.vstack((np.diag([1]*13),np.diag([-1]*13))), tc='d')
            h = matrix([2]*26, tc='d')
            b = matrix([j,1], tc='d')
            #do the optimization using QP solver
            opt = solvers.qp(P, q, G, h, A, b, options={'show_progress':False})
            w = opt['x']
            wp = np.array(w).reshape(-1,1)
        Rp = Rp + [wp.T.dot(future_return/250)[0,0]]
    R_opt['beta=%s' % j]=Rp

#R_opt = pd.DataFrame(np.array(R_opt).transpose())
#conduct the min variance with 15% target return strategy.
Rp = []
wp = np.ones((13,1))*1/13

for i in range(window, Num_days):
    future_return = R_bc[i, :].reshape(-1, 1)
    if i % alocate == 0:
        r = R_bc[(i - window):i, :]
        er = ER_bc[(i - window):i, :]
        f1 = F_bc[(i - window):i, :]
        rho = r.mean(axis=0)
        cov_f = np.cov(f1, rowvar=False)
        #run regression to get the beta
        lm = LinearRegression()
        lm.fit(f1, er)
        coeff3 = lm.coef_
        beta = coeff3[:,0]
        error = er - lm.predict(f1)
        #calculate the covariance matrix
        Q = coeff3.dot(cov_f).dot(coeff3.T)+np.diag(error.var(axis=0))
        Q_ = np.diag([1]*13)
        #preparation for the optimization
        P = matrix((Q+Lambda*Q_), tc='d')
        q = matrix(-2*Lambda*(Q_.T).dot(wp), tc='d')
        G = matrix(np.vstack((np.diag([1]*13),np.diag([-1]*13))), tc='d')
        h = matrix([2]*26, tc='d')
        A = matrix(np.vstack((rho, [1]*13)), tc='d')
        b = matrix([0.15,1], tc='d')
        #do the optimization using QP solver
        opt = solvers.qp(P, q, G, h, A, b, options={'show_progress':False})
        w = opt['x']
        wp = np.array(w).reshape(-1,1)
    Rp = Rp + [wp.T.dot(future_return/250)[0,0]]
R_opt["r=15%"]=Rp

result = pd.DataFrame(R_opt)
print(result)


# plt.plot(range(result.shape[0]),result["beta=0.5"])






# Compute PnL
pnl = PnL(result)
for i in range(5):
    plt.plot(pnl[:,i],label=i)
plt.legend(loc='best')
plt.show()

# result = R_bc.copy()/250
days = result.shape[0]

print("Last PnL after %s: " % days, PnL(result,100)[-1, :])
# Geometric mean
print("Geometric mean",gmean(result)*250)
# min
print("Daily min",np.min(result,axis=0)*250)
# max drawdown
print('max drawdown: ', MaxDrawdown(result))
# Vol
print("Volatility", Volatility(result))

# Sharp Ratio
RF = np.array(R_bc-ER_bc)[:,0].reshape(-1,1)/250
print("Sharp ratio: ", SharpRatio(result,RF))
# print("Mean sharp: ", np.mean(SharpRatio(result,RF),axis=0))

# Kurtosis
print("Kurtosis: ", Kurtosis(result))
print("Skewness: ", Skewness(result))
print("%s VaR %s days: " % (0.99, days), VaR(result, 0.99))
print("%s CVaR %s days: " % (0.99, days), CVaR(result, 0.99))

for i in range(result.shape[1]):
    # print(i)
    plt.plot((1+result[:,i]).cumprod(),label=ticker[i])
plt.legend(loc='best')
# RF = np.array(R_bc - ER_bc)[window:, 0].reshape(-1, 1) / 250
# Summary(R_bc,RF,0.99)