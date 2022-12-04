import csv
import warnings
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
from matplotlib import pyplot as plt
import lightgbm as lgb
import EO
import loaddata
warnings.filterwarnings("ignore")


feature_train,feature_test,target_train, target_test = loaddata.loaddata()

''' 利用lgb进行预测'''
#创建随机森林
n_features= feature_train.shape[1]#特征数
Model=lgb(n_estimators=10,max_features=1, max_depth=None,min_samples_split=2, bootstrap=True,random_state=0)
Model.fit(feature_train,target_train)
PredictTrain=Model.predict(feature_train)
PredictTest=Model.predict(feature_test)
MSETrain= np.sqrt(np.sum((PredictTrain - target_train)**2))/target_train.size#计算MSE
MSETest=np.sqrt(np.sum((PredictTest - target_test)**2))/target_test.size#计算MSE
print("RF训练集MSE：" +str(MSETrain))
print("RF测试集MSE："+str(MSETest))


'''目标函数(适应度函数)'''
#定义适应函数，以测试集和训练集的绝对误差和为适应度值
def fun(X):
    N_estimators =int(X[0])  #随机森林个数
    Max_depth =int(X[1])  #最大特征数
    Max_features = int(X[2])
    if (Max_features > n_features):
        Max_features = n_features
    else:
        Max_features = int(X[2])
    Model=lgb(n_estimators=N_estimators,max_features=Max_features, max_depth=Max_depth,min_samples_split=2, bootstrap=True,random_state=0)
    Model.fit(feature_train,target_train)
    PredictTrain=Model.predict(feature_train)
    PredictTest=Model.predict(feature_test)
    MSETrain= np.sqrt(np.sum((PredictTrain - target_train)**2))/target_train.size#计算MSE
    MSETest=np.sqrt(np.sum((PredictTest - target_test)**2))/target_test.size#计算MSE
    output = MSETrain+MSETest
    return output

n_features= feature_train.shape[1]#特征数
print(n_features)
#设置粒子群参数
pop = 30 #种群数量
MaxIter = 50 #最大迭代次数
dim = 3 #维度
lb = np.matrix([[1],[1],[1]]) #下边界
ub = np.matrix([[200],[30],[10]])#上边界
fobj = fun
GbestScore,GbestPositon,Curve = EO.EO(pop,dim,lb,ub,MaxIter,fobj)
print('fitness数值：',Curve)
print('最优适应度值：',GbestScore)
print('最优解：',GbestPositon)


#利用最优参数训练xgb
N_estimators = int(GbestPositon[0]) #随机森林个数
Max_depth = int(GbestPositon[1]) #最大深度
Max_features = int(GbestPositon[2])
n_features= feature_train.shape[1]#特征数
if(Max_features>n_features):
    Max_features = n_features
else:Max_features = int(GbestPositon[2])
ModelEO=lgb(n_estimators=N_estimators,max_features=Max_features, max_depth=Max_depth,min_samples_split=2, bootstrap=True,random_state=0)
ModelEO.fit(feature_train,target_train)
PredictTrainEO=ModelEO.predict(feature_train)
PredictTestEO=ModelEO.predict(feature_test)
MSETrainEO= np.sqrt(np.sum((PredictTrainEO - target_train)**2))/target_train.size#计算MSE
MSETestEO=np.sqrt(np.sum((PredictTestEO - target_test)**2))/target_test.size#计算MSE
print("RF-EO训练集MSE：" +str(MSETrainEO) )
print("RF-EO测试集MSE："+str(MSETestEO) )


#绘制适应度曲线
plt.figure(1)
plt.plot(Curve,'r-',linewidth=2)
plt.xlabel('Iteration',fontsize='medium')
plt.ylabel("Fitness",fontsize='medium')
plt.grid()
plt.title('EO-RF',fontsize='large')
plt.show()

#绘制训练集结果
# plt.figure(2)
# plt.plot(target_train,'ro-',linewidth=1,label='True value')
# plt.plot(PredictTrainEO,'b*-',linewidth=1,label='EO-RF_results')
# plt.plot(PredictTrain,'g.-',linewidth=1,label='RF_results')
# plt.xlabel('index',fontsize='medium')
# plt.ylabel("value",fontsize='medium')
# plt.grid()
# plt.title('TrainDataSet Result',fontsize='large')
# plt.legend()
# plt.show()

#绘制测试集结果
plt.figure(3)
plt.plot(target_test,'ro-',linewidth=1,label='True value ')
plt.plot(PredictTestEO,'b*-',linewidth=1,label='EO-RF_results')
plt.plot(PredictTest,'g.-',linewidth=1,label='RF_results')
plt.xlabel('index',fontsize='medium')
plt.ylabel("value",fontsize='medium')
plt.grid()
plt.title('TestDataSet Result',fontsize='large')
plt.legend()
plt.show()


# print("EO-RF训练集预测结果：" +str(PredictTrainEO))
print("EO-RF测试集预测结果：" +str(PredictTestEO))
# print("RF训练集预测结果：" +str(PredictTrain))
print("RF测试集预测结果：" +str(PredictTest))

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
print('根据真实值得输出结果是：'+str(predture))