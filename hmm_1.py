#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
# Clasificador HMM usando wavelet db4 e ICA para crisis generalizada

Created on Tue Oct 10 21:35:10 2017

@author: carlos
"""

# Cargar librerias
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import pywt
from scipy.stats import *
import scipy.io as sio
from hmmlearn.hmm import GMMHMM, GaussianHMM
from sklearn.externals import joblib

# Cargar senal
mat_contents = sio.loadmat('Patient_1/Patient_1_preictal_segment_0003.mat')

sign = mat_contents['data']
freq = mat_contents['freq']
freq = freq[0]
freq = freq[0]

W1 = sio.loadmat('W2.mat') #Matriz previamente entrenada de ICA
W = W1['W']


# Datos de la senal

Len = np.shape(sign[0,:])[0]
#x = sign[:, Len-60000:Len]
x = sign
L = np.size(x)
t = np.linspace(0, (L/freq -1/freq), L)

# Senal
sig = np.dot(W,x)
sig1 = sig[0:3, :]
plt.plot(np.transpose(sig1))
plt.show()



coeffs = pywt.swt(sig1, 'db4', level=5)

(cA5, cD5), (cA4, cD4),(cA3, cD3), (cA2, cD2), (cA1, cD1) = coeffs


#%%
dr = 50
frec = np.zeros((12, int(L/15)))
frec_n = np.zeros((12, int(L/(dr*15))))

frec[0:3,:] = cA5
frec[3:6,:] = cD5
frec[6:9,:] = cD4
frec[9:12,:] = cD3

sm = np.sum(np.abs(frec), axis=0)

#%%
for i in range(int(L/(dr*15))):
    s = frec[:, i*dr:(i+1) * dr]
    s1 = sm[i*dr:(i+1) * dr]
    for j in range(4):
        frec_n[j, i] = np.sum(s[j, :])

# Entrenar clasificador       
'''md = GaussianHMM(n_components = 7, n_iter=100).fit(np.transpose(frec_n[0:4,:]))

plt.plot(md.predict(np.transpose(frec_n[0:4,0:500])))
plt.show()'''

#print md.score(np.transpose(frec_n[0:4,:]))

# Guardar clasificador
#joblib.dump(md, "md3.pkl")

# Cargar clasificadores previamente entrenados
md1 = joblib.load("Clasificadores/md4.pkl") # preictal
md2 = joblib.load("Clasificadores/md5.pkl") # A comparar

#%%
res1 = np.zeros((2, np.shape(frec_n[0:4,:])[1]))
res2 = np.zeros_like(res1)

for i in range(15, np.shape(frec_n[0:4,:])[1]):
    res1[1, i] = md1.score(np.transpose(frec_n[0:4,i-14:i]))
    res1[0, i] = md2.score(np.transpose(frec_n[0:4,i-14:i]))
    #print i

clasif = np.argmax(res1, axis=0)
plt.plot(np.transpose(res1))
plt.show()
#plt.plot(np.transpose(res2))
plt.plot(np.argmax(res1, axis=0))
plt.show()
#alf = 0.2
#%%
ac = 0
beta = 0.01
r3 = np.zeros_like(clasif, np.float16)
for i in range(len(clasif)):
    ac = (beta) * clasif[i] + (1-beta) * ac
    r3[i] = ac
    
plt.plot(r3)
plt.show()
    
    
