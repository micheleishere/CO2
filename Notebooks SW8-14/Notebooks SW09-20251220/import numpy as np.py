import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import pandas as pd

#Fisheries reducing time step
birth_and_death_rate=0.5
b_year=-1200
x0=12230
dts=2.0**(-np.array([0,1,2,3,4,5,6]))
print(dts)
for dt in dts:
    a=1+birth_and_death_rate*dt
    b=b_year*dt
    x=np.array([x0])
    t=np.arange(0,10,dt)
    n=np.arange(0,len(t))
    for ts in range(1,len(t)): #simulation with recursive formula
        x=np.append(x,x[-1]*a+b)
    #plt.scatter(t, (x0-b/(1-a))*a**n + b/(1-a) ) #exact solution from first lesson
    plt.scatter(t, x)
a=1+birth_and_death_rate
b=b_year
plt.legend(dts)

#logistic difference
r=1
K=1000
x0=100
t_max=15
dts=2.0**(-np.array([0,1,2,3,4]))
print(dts)
for dt in dts:
    x=np.array([x0])
    t=np.arange(0,t_max,dt)
    for ts in range(1,len(t)): #simulation with recursive formula
        x=np.append(x, x[-1] + r*dt*x[-1]*(1-x[-1]/K))
    plt.scatter(t, x)
plt.legend(dts)

# playing around with different values of N0
def logistic_growth(t,N0,r,K):
    return K*N0/(N0+(K-N0)*np.exp(-r*t))
r=0.1
K=1000
N0s=np.arange(100,2000,100)
print(N0s)
t=np.linspace(0,100,num=101)
for N0 in N0s:
    plt.plot(t,logistic_growth(t,N0,r,K))

# playing around with different Values of N0
r=0.1
K=1000
N0s=[0,0.1,1,10,100]
for N0 in N0s:
    plt.plot(t,logistic_growth(t,N0,r,K))

# playing around with different values of r
rs=np.arange(0.1,1.1,0.1)
print(rs)
K=1000
N0=1
for r in rs:
    plt.plot(t,logistic_growth(t,N0,r,K))

# playing around with different Values of K
r=0.2
Ks=np.arange(100,1100,100)
print(Ks)
K=1000
N0=1
for K in Ks:
    plt.plot(t,logistic_growth(t,N0,r,K))

#logistic growth for N<<K
def logistic_growth(t,N0,r,K):
    return K*N0/(N0+(K-N0)*np.exp(-r*t))
N0=1
plt.plot(t,logistic_growth(t,N0,r,K))
plt.plot(t,N0*np.exp(r*t))
plt.ylim(0,1000)

#logistic growth for N>>K
def logistic_growth(t,N0,r,K):
    return K*N0/(N0+(K-N0)*np.exp(-r*t))
N0=10000
plt.plot(t,logistic_growth(t,N0,r,K))
plt.plot(t,1/(r/K*t+1/N0))

#Fisheries continuous limit
birth_and_death_rate=0.5
b_year=-1200
x0=12230
dts=2.0**(-np.array([0,1,2,3,4]))
print(dts)
for dt in dts:
    a=1+birth_and_death_rate*dt
    b=b_year*dt
    x=np.array([x0])
    t=np.arange(0,5,dt)
    n=np.arange(0,len(t))
    for ts in range(1,len(t)): #simulation with recursive formula
        x=np.append(x,x[-1]*a+b)
    #plt.scatter(t, (x0-b/(1-a))*a**n + b/(1-a) ) #exact solution from first lesson
    plt.scatter(t, x)
a=1+birth_and_death_rate
b=b_year
t=np.linspace(0,5,1001)
plt.plot(t, (x0-b/(1-a))*np.exp(-(1-a)*t) + b/(1-a) )

#allometric growth 
mammals = pd.read_csv("mammals.csv")
plt.scatter(mammals['body'],mammals['brain'])

#allometric growth log-log-scale
plt.scatter(mammals['body'],mammals['brain'])
plt.yscale('log')
plt.xscale('log')

#linear regression in log-log-plot
from scipy.stats import linregress
regr = linregress(np.log(mammals['body']), np.log(mammals['brain']))
#relative grow rate of brain is 0.75 relative grow rate of the body
print(regr)

plt.scatter(mammals['body'],mammals['brain'])
plt.yscale('log')
plt.xscale('log')
x=10**np.arange(-3,4,0.01)
plt.plot(x, x**regr[0]*np.exp(regr[1]))

#Commpertz growth
def compertz_growth(t,V0,k,alpha):
    return V0*np.exp(k/alpha*(1-np.exp(-alpha*t)))
alpha=0.1
k=0.1
V0s=np.linspace(0,2,num=10)
t=np.linspace(0,100,num=10001)
for V0 in V0s:
    plt.plot(t,compertz_growth(t,V0,k,alpha))