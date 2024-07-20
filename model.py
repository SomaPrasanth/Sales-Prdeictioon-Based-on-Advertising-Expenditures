import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt

def computeError(x,y,w,b):
    err=0
    m=x.shape[0]
    for i in range(m):
        f=np.dot(x[i],w)+b
        err+=(f-y[i])**2
    err/=(2*m)
    return err

def compute_cost(x,y,w,b,lam):
    m,n=x.shape
    cost=0
    err_term=0
    regular_term=0
    for i in range(m):
        f=np.dot(x[i],w)+b
        err_term+=(f-y[i])**2
    err_term/=(2*m)
    for i in range(n):
        regular_term+=(w[i])**2
    regular_term*=(lam/(2*m))

    cost=err_term+regular_term
    return cost

def compute_derivatives(x,y,w,b,lam):
    m,n=x.shape
    dj_dw=np.zeros(n)
    dj_db=0
    for i in range(m):
        f=np.dot(x[i],w)+b
        error=f-y[i]
        for j in range(n):
            dj_dw[j]+=error*x[i, j]
        dj_db+=error
    dj_dw=(dj_dw/m)+(lam/m)*w
    dj_db/=m
    return dj_dw, dj_db


def coumpute_gradient(x,y,w,b,alpha,itertions,lam):
    for i in range(itertions):    
        dj_dw,dj_db=compute_derivatives(x,y,w,b,lam)
        w=w-(alpha*dj_dw)
        b=b-(alpha*dj_db)  
        if i%100==0:
                print(f'Cost at itertion-{i} is {compute_cost(x,y,w,b,lam)}')              
    return w,b   

def predcit(x,w,b):
    result=np.dot(x,w)+b
    return result

df=pd.read_csv('TrainingSet.csv')
a0=list(df.iloc[:,0])
a1=list(df.iloc[:,1])
a2=list(df.iloc[:,2])
X_list=[]
Y_train=np.array(df.iloc[:,3])
for i in range(len(a0)):
    temp=[a0[i],a1[i],a2[i]]
    X_list.append(temp)
X_train=np.array(X_list)


df=pd.read_csv('CrossValidationSet.csv')
a0=list(df.iloc[:,0])
a1=list(df.iloc[:,1])
a2=list(df.iloc[:,2])
X_list=[]
Y_CV=np.array(df.iloc[:,3])
for i in range(len(a0)):
    temp=[a0[i],a1[i],a2[i]]
    X_list.append(temp)
X_CV=np.array(X_list)

w=np.zeros(3)
b=0.0
iterations=10000
alpha=0.00004
lam=0.5
w,b=coumpute_gradient(X_train,Y_train,w,b,alpha,iterations,lam)
m_train=X_train.shape[0]
m_CV=X_CV.shape[0]

print(f'\nTraning Set error:{computeError(X_train,Y_train,w,b,)}')
print(f'Cross Validation Set error:{computeError(X_CV,Y_CV,w,b,)}')

yt=float(input("\nEnter Advertising Expdeniture on Youtube (in $1000):"))
fb=float(input("Enter Advertising Expdeniture on Facebook (in $1000):"))
ns=float(input("Enter Advertising Expdeniture on Newspaper (in $1000):"))
x_sample=np.array([yt,fb,ns])
print(f"Sales(in $1000):{predcit(x_sample,w,b)}")

yt=float(input("\nEnter Advertising Expdeniture on Youtube (in $1000):"))
fb=float(input("Enter Advertising Expdeniture on Facebook (in $1000):"))
ns=float(input("Enter Advertising Expdeniture on Newspaper (in $1000):"))
x_sample=np.array([yt,fb,ns])
print(f"Sales(in $1000):{predcit(x_sample,w,b)}")


# 245.4,5.2,43.0,14.8
# 35.6,32.8,54.4,11.1