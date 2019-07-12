# -*- coding: utf-8 -*-
"""
Created on Sat Jun 23 23:28:35 2018
@author: Leilei Cao
"""
from __future__ import division
import numpy as np
import math
from sklearn.svm import SVR
import matplotlib.pyplot as plt
import time

time_start=time.time()
runs=1
N=100  # population size
Gen=1000 # number of generations
tao=10
nt=10
od=4  # orders of variables
d=10  # dimensions of variables
Lb=0  # lower range of variables
Ub=1   # upper range of variables
p=0.8                
F=0.5  # scaled factor
CR=0.5  # corssover rate 
T=20
   
def DF1(x,k):
	G=abs(np.sin(0.5*math.pi*k))
	H=0.75*np.sin(0.5*math.pi*k)+1.25
	g=1+sum(np.square(x[1:]-G))
	f=np.zeros((2))
	f[0]=x[0]
	f[1]=g*(1-math.pow((x[0]/g),H))
	return f
 
def decom(x,tk,lmta,idp):
    fit=DF1(x,tk)
    f=np.zeros((2))
    f[0]=lmta[0]*np.abs(fit[0]-idp[0])
    f[1]=lmta[1]*np.abs(fit[1]-idp[1])
    return max(f)

def mani_d(X,Y):
    x_X=X.shape[0]
    x_Y=Y.shape[0]
    dis=np.zeros((x_X,x_Y))
    for i in range(x_X):
        for j in range(x_Y):
            dis[i][j]=np.linalg.norm(X[i,:]-Y[j,:],ord=2)
    dDistance=np.sum(dis.min(axis=1))/x_X
    return dDistance

#initilize weights
weight=np.zeros((N,2))
for i in range(N):
    weight[i][0]=i/N
    weight[i][1]=1-i/N
neighbor=np.zeros((N,N),int)
Distance=np.zeros((N,N))
for i in range(N):
    for j in range(i+1,N):
        Distance[i][j]=np.linalg.norm(weight[i]-weight[j],ord=2) # compute E-distance of each pair of weights
        Distance[j][i]=Distance[i][j]
    index1=np.argsort(Distance[i])
    neighbor[i]=index1    # neighbors of each weight

#initilize the population
migd=np.zeros((runs))
for run in range(runs):    
    sol=np.zeros((N,d))
    fitness=np.zeros((N,2))
    for i in range(N):
        sol[i]=Lb+(Ub-Lb)*np.random.random(d)
        fitness[i]=DF1(sol[i],0)
    Z=np.zeros((2))
    Z=fitness.min(0)

    Fit=np.zeros((int(Gen/tao),N,2))
    all_s=np.zeros((int(Gen/tao),N,d))

    for t in range(1,Gen+1):
        K=math.ceil(t/tao)-1
        tk=1/nt*math.floor((t-1)/tao)
        if t>tao and t%tao==1:
            Fit[K-1,...]=fitness
            all_s[K-1,...]=sol
            
            if t>(od+1)*tao and t%tao==1:
                for i in range(N):
                    for j in range(d):
                        train=[]
                        for jj in range(len(all_s[0:K,i,j])-od):
                            train.append(all_s[jj:jj+od+1,i,j])
                        train=np.reshape(train,(K-od,od+1))
                        x_train=train[:,:-1]
                        y_train=train[:,-1]
                        svr=SVR(kernel='rbf', epsilon=0.05, C=1e3)
                        sol[i,j]=svr.fit(x_train,y_train).predict(all_s[K-od:K,i,j].reshape(-1,od))
                        sol[i,j]=max(min(sol[i,j],Ub),Lb)

            for i in range(N):
                fitness[i]=DF1(sol[i],tk)
            Z=fitness.min(0)
        
        for i in range(N):
            if np.random.random()<p:
                P=neighbor[i,:T]
            else:
                P=list(range(N))
                
            xx=np.random.choice(P)
            yy=np.random.choice(P)
            zz=np.random.choice(P)
            V=np.zeros((d))
            for j in range(d):
                if np.random.random()<CR:
                    V[j]=sol[xx,j]+F*(sol[yy,j]-sol[zz,j])
                else:
                    V[j]=sol[i,j]
                if np.random.random()<0.5:
                    delta=math.pow(2*np.random.random(),1/21)-1
                else:
                    delta=1-math.pow((2-2*np.random.random()),1/21)
                if np.random.random()<(1/d):
                    V[j]=V[j]+delta*(Ub-Lb)
            for ii in range(d):
                V[ii]=max(min(V[ii],Ub),Lb)
            Fitness=DF1(V,tk)
            if Fitness[0]<Z[0]:
                Z[0]=Fitness[0]
            if Fitness[1]<Z[1]:
                Z[1]=Fitness[1]
            
            for j in range(len(P)):
                gg1=decom(sol[P[j]],tk,weight[P[j]],Z)
                gg2=decom(V,tk,weight[P[j]],Z)
                if gg2<gg1:
                    sol[P[j]]=V
                    fitness[P[j]]=Fitness
                 
    Fit[K,...]=fitness
    all_s[K,...]=sol

    # IGD values
    x=np.zeros((500,d))
    igd=np.zeros((int(Gen/tao)))
    for tt in range(1,int(Gen/tao)+1):
        tf=np.zeros((500,2))
        D=np.zeros((500,N))
        min_D=np.zeros((500))
        for i in range(500):
            x[i][0]=i/499
            x[i,1:]=abs(np.sin(0.5*math.pi*(tt-1)/nt))
            tf[i]=DF1(x[i],(tt-1)/nt)
            for j in range(N):
                D[i][j]=np.linalg.norm(tf[i]-Fit[tt-1,j,:],ord=2)
            min_D[i]=min(D[i])

        igd[tt-1]=np.sum(min_D)/500
        plt.plot(tf[:,0],tf[:,1])
    migd[run]=np.mean(igd)
    print('we have completed run:%d'%(run))
aver_migd=np.mean(migd)
std_migd=np.std(migd)

plt.figure()
plt.plot(igd,marker='o')   
plt.show()

time_end=time.time()
print ('total running time:',(time_end-time_start)/60)
print (migd)    
    
