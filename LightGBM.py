import csv
import warnings
import pandas as pd
from sklearn.metrics import mean_squared_error as mse
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold
import lightgbm as lgb
import EO
import loaddata
warnings.filterwarnings("ignore")


feature_train,feature_test,target_train, target_test = loaddata.loaddata()

''' Using lgb to predict'''

Model=lgb.LGBMRegressor()
Model.fit(feature_train,target_train)
PredictTrain=Model.predict(feature_train)
PredictTest=Model.predict(feature_test)
MSETrain= mse(PredictTrain,target_train)
MSETest=mse(PredictTest,target_test)
print("MSE of LightGBM training set:" +str(MSETrain))
print("MSE of LightGBM testing set:"+str(MSETest))


'''Objective function (fitness function)'''
#The fitness function is defined, and the absolute error sum of the test set and the training set is the fitness value.
def fun(X):
    N_estimators =int(X[0]) 
    Max_depth =int(X[1])  
    lr = float(X[2])
    Sub = float(X[3])
    Model=lgb.LGBMRegressor(n_estimators=N_estimators, max_depth=Max_depth, learning_rate=lr, subsample=Sub)
    Model.fit(feature_train,target_train)
    PredictTrain=Model.predict(feature_train)
    return mse(PredictTrain,target_train)


#Set the parameters of the optimizer
pop = 30 
MaxIter = 100 
dim = 4 
lb = np.matrix([[100],[3],[0.001],[0.5]]) 
ub = np.matrix([[200],[15],[0.1],[1.0]])
fobj = fun
GbestScore,GbestPositon,Curve = EO.EO(pop,dim,lb,ub,MaxIter,fobj)
print('best solve:',GbestPositon)


#Using the optimal parameters to train xgb
N_estimators = int(GbestPositon[0])
Max_depth = int(GbestPositon[1])
lr = float(GbestPositon[2])
Sub = float(GbestPositon[3]) 
ModelEO=lgb.LGBMRegressor(n_estimators=N_estimators, max_depth=Max_depth, learning_rate=lr, subsample=Sub)
ModelEO.fit(feature_train,target_train)
PredictTrainEO=ModelEO.predict(feature_train)
PredictTestEO=ModelEO.predict(feature_test)
MSETrainEO= mse(PredictTrainEO,target_train)
MSETestEO=mse(PredictTestEO,target_test)
print("MSE of EO-LightGBM training set:" +str(MSETrainEO))
print("MSE of EO-LightGBM testing set:"+str(MSETestEO))


#Draw fitness curve
plt.figure(1)
plt.plot(Curve,'r-',linewidth=2)
plt.xlabel('Iteration',fontsize='medium')
plt.ylabel("Fitness",fontsize='medium')
plt.grid()
plt.title('EO-RF',fontsize='large')
plt.show()


#Draw the testing set results
plt.figure(3)
plt.plot(target_test,'ro-',linewidth=1,label='True value ')
plt.plot(PredictTestEO,'b*-',linewidth=1,label='EO-LightGBM_results')
plt.plot(PredictTest,'g.-',linewidth=1,label='LightGBM_results')
plt.xlabel('index',fontsize='medium')
plt.ylabel("value",fontsize='medium')
plt.grid()
plt.title('TestDataSet Result',fontsize='large')
plt.legend()
plt.show()

#Prediction of actual engineering parameters.
predture = ModelEO.predict([[11.67,9.8,13.11],
                             [8.26,13.5,13.93],
                             [17.4,12.54,13.19],
                             [18.7,12.55,15.6],
                             [25.49,14.77,15.27],
                             [13.7,13.06,17.95],
                             [10.37,13.03,14.26],
                             [10.03,13.37,15.11],
                             [8.45,9.91,12.04],
                             [9.06,10.91,11.38]])
print('According to the real value, the output results are:'+str(predture))