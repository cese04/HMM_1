# -*- coding: utf-8 -*-
"""
Created on Wed Nov 08 02:07:00 2017

@author: CarlosEmiliano
"""

from __future__ import division
import numpy as np
from scipy.stats import *
import scipy.io as sio
import pandas as pd
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
from scipy import signal

#mat_contents = sio.loadmat('3 Paciente 3 M.F.E/3.MAS_FRANCO_AVG.mat')
#mat_contents = sio.loadmat('2 Paciente 2 G.G.A/12.GONZALEZ_GOMEZ_ALBERTO.mat')
mat_contents = sio.loadmat('1 Paciente 1 A.B/1.GOMEZ_TELLO_ALAN.mat')


mat2 = sio.loadmat('Matrices/P1.mat')
mat = np.transpose(mat2['crisis'])

sign = mat_contents['senal']
#freq = mat_contents['freq']
#freq = freq[0]
#freq = freq[0]
freq = 200
L = np.size(mat[0, :])

x = np.transpose(sign[:,0:19])
t_k = np.zeros_like(x)
for i in range(1,len(x[0,:])-1):
    t_k[:,i] = ((x[:, i]) ** 2 - (x[:, i-1]) * (x[:,i+1]))
    
dr = 200
dt = np.zeros((mat.shape[0], int(mat.shape[1])))

for i in range(int(L)):
    dt[:,i] = np.sum(np.abs(t_k[:, i * dr:(i + 1) * dr]), axis=1)
    
    
b, a = signal.butter(8, 0.125) #octavo orden y 0.125 frecuancia de Nyquist
y = signal.filtfilt(b, a, dt, padlen=150)
#%%
'''plt.imshow(x, extent=[0,1,0,1])
plt.autoscale(tight=True)
plt.show()'''


'''thr = 0.25*np.std(t_k)
plt.imshow(np.abs(t_k)>thr, extent=[0,1,0,1])
plt.imshow(np.abs(t_k)>thr, extent=[0,1,0,1])
plt.autoscale(tight=True)
plt.show()'''

#thr = 0.*np.std(t_k)
plt.imshow(np.transpose(mat)>10, extent=[0,1,0,1])
#plt.imshow(np.abs(t_k)>thr, extent=[0,1,0,1])
plt.autoscale(tight=True)
plt.show()


thr = 10*np.std(t_k)
plt.imshow(np.transpose(np.abs(dt))>thr, extent=[0,1,0,1])
plt.autoscale(tight=True)
plt.show()