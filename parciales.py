# -*- coding: utf-8 -*-
"""
# Clasificador para pacientes con epilepsia parcial usando datos estadisticos

Created on Tue Nov 14 01:46:56 2017

@author: CarlosEmiliano
"""

from __future__ import division
import numpy as np
from scipy.stats import *
import scipy.io as sio
import matplotlib.pyplot as plt
from hmmlearn.hmm import GaussianHMM 
from sklearn.externals import joblib
import warnings

warnings.filterwarnings("ignore")


#%% Carga senal y datos
mat_contents = sio.loadmat('1 Paciente 1 A.B/1.GOMEZ_TELLO_ALAN.mat')
# mat_contents = sio.loadmat('2 Paciente 2 G.G.A/12.GONZALEZ_GOMEZ_ALBERTO.mat')
sign = mat_contents['senal']

#

fs = 200

# Selcciona canal para detectar
canal = input('Seleccionar canal: ')

canal = canal - 1
dr = 200

x = sign[:, canal]
L = np.size(x)
t = np.linspace(0, (L/fs - 1/fs), L)

plt.plot(t, x)
plt.show()

#%% Extraccion de caracteristicas
# Calculo Teager-Kaiser
t_k = np.zeros_like(x)
for i in range(1, len(x)-1):
    t_k[i] = np.abs((np.abs(x[i]) ** 2 - np.abs(x[i-1]) * np.abs(x[i+1])))


dt = np.zeros((4, int(L/dr)))

# Ventaneo y calculo de caracteristicas
for i in range(int(L/dr)):
    s = x[i * dr:(i + 1) * dr]
    s = s.ravel()
    dt[0, i] = entropy(s**2)
    dt[1, i] = np.sum(t_k[i*dr:(i+1)*dr])
    dt[2, i] = skew(s)
    dt[3, i] = kurtosis(s)

# Mostrar caracteristicas
L2 = np.shape(dt[0, :])[0]
plt.subplot(511)
plt.plot(np.arange(L2), dt[0, :])

plt.subplot(512)
plt.plot(np.arange(L2), dt[1, :])

plt.subplot(513)
plt.plot(np.arange(L2), dt[2, :])

plt.subplot(514)
plt.plot(np.arange(L2), dt[3, :])

plt.show()
#%% Entrenar
    
y1 = dt[:, 0:15]
y2 = dt[:, 200:400]
y3 = dt[:, 1700:1900]
y31 =np.append(y1, y2, axis=1)
y4 = np.append(y31, y3, axis=1)

md = GaussianHMM(n_components=7, n_iter=100).fit(np.transpose(y4))

print md.score(np.transpose(y4))
plt.plot(md.predict(np.transpose(y4)))
plt.show()

##joblib.dump(md, "Clasificadores/md12.pkl")

#%% Evaluar

md1 = joblib.load("Clasificadores/md11.pkl")     #Preictal
md2 = joblib.load("Clasificadores/md12.pkl")     #Interictal
#md3 = joblib.load("Clasificadores/md8.pkl")     #Ictal


res1 = np.zeros((2, np.shape(dt)[1]))
#res2 = np.zeros_like(res1)

for i in range(20, np.shape(dt)[1]):
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




