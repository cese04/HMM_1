# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 13:15:39 2017

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
    
    s = x[:,i*dr:(i+1)*dr]
    
    s = s.ravel()
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

#%% Evaluar

md1 = joblib.load("Clasificadores/md7.pkl")     #Preictal
md2 = joblib.load("Clasificadores/md6.pkl")     #Interictal
md3 = joblib.load("Clasificadores/md8.pkl")     #Ictal


res1 = np.zeros((2, np.shape(y)[1]))
res2 = np.zeros_like(res1)

for i in range(20, np.shape(y)[1]):
    #res1[1, i] = md1.score(np.transpose(dt[:,i-50:i]))
    res1[1, i] = md1.score(np.transpose(dt[:,i-20:i]))
    res1[0, i] = md2.score(np.transpose(dt[:,i-20:i]))
    #print i

clasif = np.argmax(res1, axis=0)

ac = 0
beta = 0.05
r3 = np.zeros_like(clasif, np.float16)
for i in range(len(clasif)):
    ac = (beta) * clasif[i] + (1-beta) * ac
    r3[i] = ac
    
plt.plot(r3)
plt.show()
    
