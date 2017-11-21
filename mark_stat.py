# -*- coding: utf-8 -*-
"""
Created on Wed Nov 08 01:07:41 2017

@author: CarlosEmiliano
"""

from __future__ import division
import numpy as np
from scipy.stats import *
import scipy.io as sio
import pandas as pd
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
from hmmlearn.hmm import GMMHMM, GaussianHMM 
from scipy import signal
from sklearn.externals import joblib

mat_contents = sio.loadmat('3 Paciente 3 M.F.E/3.MAS_FRANCO_AVG.mat')

sign = mat_contents['senal']
#freq = mat_contents['freq']
#freq = freq[0]
#freq = freq[0]
freq = 200

W1 = sio.loadmat('WP3.mat') #Matriz previamente entrenada de ICA
W = W1['W']


x = np.transpose(sign[:,0:19])
L = np.size(x[0,:])
t = np.linspace(0, (L/freq -1/freq), L)

print np.shape(x)
dr = 200
sig = np.dot(W,x)
x = sig[0:5, :]


# Calculo Teager-Kaiser
t_k = np.zeros_like(x)
for i in range(1,len(x[0,:])-1):
    t_k[:,i] = np.abs((np.abs(x[:,i]) ** 2 - np.abs(x[:,i-1]) * np.abs(x[:,i+1])))


dt = np.zeros((4,int(L/dr)))

# Ventaneo y calculo de caracteristicas
for i in range(int(L/dr)):
    #s = x[i:i+dr]
    s = x[:,i*dr:(i+1)*dr]
    #s = np.reshape(s, (1,np.product(s.shape)))
    s = s.ravel()
    #print np.shape(s)
    #print skew(s)
    #skw.append(skew(s))
    #krt.append(kurtosis(s))
    dt[0,i] = entropy(s**2)
    #print ent[i]
    dt[1,i] = np.sum(t_k[:,i*dr:(i+1)*dr])
    dt[2,i] = skew(s)
    dt[3,i] = kurtosis(s)

dt1 = np.transpose(dt)
bc = []

L2 = np.shape(dt[0,:])[0]
plt.subplot(511)
plt.plot(np.arange(L2), dt[0,:])

plt.subplot(512)
plt.plot(np.arange(L2), dt[1,:])

plt.subplot(513)
plt.plot(np.arange(L2), dt[2,:])

plt.subplot(514)
plt.plot(np.arange(L2), dt[3,:])

plt.show()

#plt.subplot(515)
#plt.plot(np.arange(L2), frec_n[4,:])
#plt.title('cD1')
# Filtrado de las caracteristicas para suavizar

b, a = signal.butter(8, 0.125) #octavo orden y 0.125 frecuancia de Nyquist
y = signal.filtfilt(b, a, dt, padlen=150)


plt.subplot(511)
plt.plot(np.arange(L2), y[0,:])

plt.subplot(512)
plt.plot(np.arange(L2), y[1,:])

plt.subplot(513)
plt.plot(np.arange(L2), y[2,:])

plt.subplot(514)
plt.plot(np.arange(L2), y[3,:])

plt.show()

'''for i in range(3,30):
    km = GaussianMixture(n_components = i, covariance_type = 'diag').fit(dt1)
    bc.append(km.bic(dt1))
    
plt.plot(bc)
plt.show()


vec = km.predict(dt1)

plt.scatter(dt[0,:], dt[1,:], c=vec)
plt.show()

print km.bic(dt1)'''

#%%
#y1 = dt[:,700:1500]
#y2 = dt[:,1600:2000]
#y3 =np.append(y1,y2, axis=1)
y3 = dt[:,220:270]

md = GaussianHMM(n_components = 7, n_iter=100).fit(np.transpose(y3))

print md.score(np.transpose(y3))
plt.plot(md.predict(dt1))
plt.show()

#joblib.dump(md, "Clasificadores/md7.pkl")
