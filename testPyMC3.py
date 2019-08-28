import numpy as np
import math
#import theano
import theano.tensor as tt
#from theano.compile.ops import as_p
import pymc3 as pm
import scipy.io as sio

#import seaborn as sns
#import matplotlib.pyplot as plt

m=1862
n=201
d=120

def readH(path):
    dummy=sio.loadmat(path+'Trans.mat')
    H=dummy['H']
    return H

def readU(path,seg_index=12,exc_index=71):
    TrainData = sio.loadmat(path + 'TmpSeg' + str(seg_index) + 'exc' + str(exc_index) + '.mat')
    U = (TrainData['U'])
    return U.transpose()
def genNoisy(Y,noisevar=1e-4,index=2):
    (a,b)=Y.shape
    noise=np.random.normal(0, math.sqrt(noisevar), [a,b])
    return Y+noise

#@as_op(itypes=[tt.dvector], otypes=[tt.dvector])
def decodeMean(z):
    return z

def decodeVar(z):
    return z * 100
pathH = '/Users/sg9872/Desktop/Research/Data/Halifax-EC/Simulation/1862/Input/'
pathU = '/Users/sg9872/Desktop/Research_Projects/Sequence_VAE/BigData/'
H = readH(pathH)
print(H.shape)
U = readU(pathU)
Y = genNoisy(np.matmul(H, U))
print('Size of Y:',Y.shape)
beta = 1e5
covY=(1/beta)*np.identity(d)

with pm.Model() as BayesNet:
    z=pm.Normal('z',mu=0,sd=0.001,shape=1862)

    meanX=pm.Deterministic('meanX',decodeMean(z))
    print('Size of meanX',meanX.shape)
    precX=pm.Deterministic('precX',decodeVar(z))
    x = pm.Normal('x', mu=meanX, tau=precX, shape=1862) # good

    y=pm.MvNormal('y',mu=tt.dot(H,x),cov=covY,observed=Y[:,1])
    trace=pm.sample(100,tune=50)
