# -*- coding: utf-8 -*-
"""
Created on Fri Apr  1 16:15:33 2022

@author: Oleg
"""
import numpy as np
from math import sqrt
import scipy.stats as sps
import random
import matplotlib.pyplot as plt

N = 50000 
M = N/100 
sig = 0.5
dt = 0.05 
R=1.5432e-06
R1=0.0000000001
Rd=R/dt
Rd1=R1/dt
H=np.array([[1,0]])

for i in range(0,1):
    v = np.empty((1,N))
    x1 = np.empty((i+1,N))
    y1_f = np.empty((i+1,N))
    omega=0.6
    alf = 0.015
    bet = sqrt(omega**2-alf**2)
    F =np.array([[0,1],[-(alf**2 + bet**2), -2*alf]]) 
    G =np.array([[0], [sig*sqrt(4*alf*(alf**2 + bet**2))]])
    Fi = np.eye(2) + F*dt
    Gam = G*dt
    P = np.array([[sig**2,0],[0,(sig**2)*(alf**2 + bet**2)]])
    w = sps.norm(loc=0, scale=1/sqrt(dt)).rvs(size=[1,N])
    x = np.matmul(np.sqrt(P),sps.norm(loc=0, scale=1).rvs(size=[2,1]))
    
    for k in range (0,N):
        v[0][k]=sqrt(Rd)*random.random()
        x = np.matmul(Fi,x) + Gam*w[0][k]
        x1[i][k]=np.matmul(H,x)
        y1_f[i][k] = np.matmul(H,x)+v[0][k]
t=np.array(range(1,5000))
plt.plot(t,y1_f[0][1:5000])
