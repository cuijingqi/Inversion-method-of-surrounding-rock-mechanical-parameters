import numpy as np
import random
import math
import copy
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

''' Population initialization function '''


def initial(pop, dim, ub, lb):
    X = np.zeros([pop, dim])
    for i in range(pop):
        for j in range(dim):
            X[i, j] = random.random() * (ub[j] - lb[j]) + lb[j]

    return X, lb, ub


'''Boundary check function'''


def BorderCheck(X, ub, lb, pop, dim):
    for i in range(pop):
        for j in range(dim):
            if X[i, j] > ub[j]:
                X[i, j] = ub[j]
            elif X[i, j] < lb[j]:
                X[i, j] = lb[j]
    return X


'''Calculate fitness function'''


def CaculateFitness(X, fun):
    pop = X.shape[0]
    fitness = np.zeros([pop, 1])
    for i in range(pop):
        fitness[i] = fun(X[i, :])
    return fitness


'''adaptation sequence'''


def SortFitness(Fit):
    fitness = np.sort(Fit, axis=0)
    index = np.argsort(Fit, axis=0)
    return fitness, index


'''Sort locations according to fitness'''


def SortPosition(X, index):
    Xnew = np.zeros(X.shape)
    for i in range(X.shape[0]):
        Xnew[i, :] = X[index[i], :]
    return Xnew


'''Main'''
def EO(pop,dim,lb,ub,MaxIter,fun):
    V = 1
    a1=2
    a2=1
    GP=0.5
    X,lb,ub = initial(pop, dim, ub, lb) #initialization population
    fitness = CaculateFitness(X,fun) #Calculate the fitness value
    fitness,sortIndex = SortFitness(fitness) #Sorting the fitness values
    X = SortPosition(X,sortIndex) #individual sorting
    GbestScore = copy.copy(fitness[0])
    GbestPositon = copy.copy(X[0,:])
    Curve = np.zeros([MaxIter,1])
    Xnew = copy.copy(X)   
    C_pool= np.zeros([5,dim]) #balance tank
    for t in range(MaxIter):
        print("iteration times:" + str(t))
        T = (1-t/MaxIter)**(a2*t/MaxIter) #inertia weight factor
        Ceq1 = copy.copy(X[0,:])
        Ceq2 = copy.copy(X[1,:])
        Ceq3 = copy.copy(X[2,:])
        Ceq4 = copy.copy(X[3,:])    
        #Calculate the average solution
        Ceq_ave=(Ceq1+Ceq2+Ceq3+Ceq4)/4
        C_pool[0,:] = copy.copy(Ceq1)
        C_pool[1,:] = copy.copy(Ceq2)
        C_pool[2,:] = copy.copy(Ceq3)
        C_pool[3,:] = copy.copy(Ceq4)
        C_pool[4,:] = copy.copy(Ceq_ave)
        
        for i in range(pop):
            Lambda = np.random.random([1,dim])
            r = np.random.random([1,dim])
            randIndex = np.random.randint(4) #Randomly select from the pool
            Ceq = copy.copy(C_pool[randIndex,:])
            F=a1*np.sign(r-0.5)*(np.exp(-Lambda*T)-1) #Calculate the index coefficient F
            r1 = np.random.random()
            r2 = np.random.random()
            GCP = 0.5*r1*np.ones([1,dim])*(int(r2>=GP)) #Calculation of mass generation rate G
            G0 = GCP*(Ceq - Lambda*X[i,:])
            G = G0*F
            Xnew[i,:] = Ceq + (X[i,:] - Ceq)*F+(G/Lambda*V)*(1-F) #types of regeneration site
            
        Xnew = BorderCheck(Xnew,ub,lb,pop,dim)
        fitnew = CaculateFitness(Xnew,fun)
        for i in range(pop):
            if fitnew[i]<fitness[i]:
                X[i,:] = copy.copy(Xnew[i,:])
                fitness[i] = copy.copy(fitnew[i])
        
        fitness,sortIndex = SortFitness(fitness) #Sorting the fitness values
        X = SortPosition(X,sortIndex) #individual sorting
        if(fitness[0]<=GbestScore): #Update global optimum
            GbestScore = copy.copy(fitness[0])
            GbestPositon = copy.copy(X[0,:])
        Curve[t] = GbestScore
    
    return GbestScore,GbestPositon,Curve
    
