import pandas as pd
from pandas import Series, DataFrame
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
sample_length = 2000
sample_length2 = sample_length*2
xad = np.zeros((sample_length ,6))
xaq = np.zeros((sample_length ,6))
ybd = np.zeros((sample_length ,1))
ybq = np.zeros((sample_length ,1))
xall = np.zeros((sample_length2,6))
yall = np.zeros((sample_length2,1))
ad1 = pd.read_csv('ad1.csv')
ad2 = pd.read_csv('ad2.csv')
ad3 = pd.read_csv('ad3.csv')
ad4 = pd.read_csv('ad4.csv')
ad5 = pd.read_csv('ad5.csv')
ad6 = pd.read_csv('ad6.csv')
adall = pd.concat([ad1,ad2,ad3,ad4,ad5,ad6], axis =1)
adall.columns = ['ad1','ad2','ad3','ad4','ad5','ad6']
aq1 = pd.read_csv('aq1.csv')
aq2 = pd.read_csv('aq2.csv')
aq3 = pd.read_csv('aq3.csv')
aq4 = pd.read_csv('aq4.csv')
aq5 = pd.read_csv('aq5.csv')
aq6 = pd.read_csv('aq6.csv')
aqall = pd.concat([aq1,aq2,aq3,aq4,aq5,aq6], axis =1)
aqall.columns = ['aq1','aq2','aq3','aq4','aq5','aq6']
bd = pd.read_csv('bd.csv')
bd.columns = ['bd']
bq = pd.read_csv('bq.csv')
bq.columns = ['bq']
adallmean = adall.mean()
aqallmean = aqall.mean()
x = np.array([adallmean,aqallmean])

bdmean = bd.mean()
bqmean = bq.mean()
y = np.array([bdmean,bqmean])
for j in range(sample_length):
    k = 2*j
    g = 2*j+1
    xad[j] = np.array([adall.ix[j]])
    xaq[j] = np.array([aqall.ix[j]])
    ybd[j] = np.array([bd.ix[j]])
    ybq[j] = np.array([bq.ix[j]])
    xall[k] = xad[j]
    xall[g] = xaq[j]
    yall[k] = ybd[j]
    yall[g] = ybq[j]
regr = linear_model.LinearRegression()
regr.fit(xall, yall)
#regr.fit(x, y)
print('Coefficients: \n', regr.coef_)