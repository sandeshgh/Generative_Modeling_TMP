from __future__ import print_function

import torch
import torch.utils.data
from torch import nn, optim
from torch.autograd import Variable

import time
#import theano
#import theano.tensor as tt

import numpy as np
import math
import pymc
from pymc import MvNormal
from pymc import deterministic
import scipy.io as sio


t0=time.time()
#import seaborn as sns
#import matplotlib.pyplot as plt
seq_length = 201
input_dim = 1862

d=120

class SeqVaeFull(nn.Module):
    def __init__(self):
        super(SeqVaeFull, self).__init__()

        self.fc1 = nn.LSTM(input_dim, 800)
        self.fc21 = nn.LSTM(800, 50)
        self.fc22 = nn.LSTM(800, 50)
        self.fc3 = nn.LSTM(50, 800)
        self.fc41 = nn.LSTM(800, input_dim)
        self.fc42 = nn.LSTM(800, input_dim)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()


    def encode(self, x):
        out, hidden=self.fc1(x)
        h1 = self.relu(out)
        out21,hidden21=self.fc21(h1)
        out22, hidden22 = self.fc22(h1)
        return out21, out22

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z):
        out3,hidden3=self.fc3(z)
        h3 = self.relu(out3)
        out1,hidden1=self.fc41(h3)
        out2, hidden2 = self.fc42(h3)
        return (out1), (out2)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        muTheta,logvarTheta=self.decode(z)
        return muTheta,logvarTheta, mu, logvar


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
def Posterior(MuZ,SigmaZ,H,Y,beta):
    (M,N)=MuZ.shape
    betaHH=beta*np.matmul(H.transpose(),H)
    PrecisionZ=np.reciprocal(SigmaZ)
    B=beta*np.matmul(H.transpose(),Y)+np.multiply(PrecisionZ,MuZ)
    logdetP=float(0)
    MeanU=np.empty(shape=[M,0])
    for i in range(N):
        PreZ=(PrecisionZ[:,i])
        PrecisionU=betaHH+np.diag(PreZ)
        Ui=np.linalg.solve(PrecisionU,B[:,i])
        Ui.shape=(M,1)
        #print('Ui is ',Ui)
        #print(Ui)
        MeanU=np.append(MeanU,Ui,axis=1)
        (s,logdet)=np.linalg.slogdet(PrecisionU)
        logdetP=logdetP+logdet
    return MeanU,0.5*logdetP


pathH = '/Users/sg9872/Desktop/Research/Data/Halifax-EC/Simulation/1862/Input/'
pathU = '/Users/sg9872/Desktop/Research_Projects/Sequence_VAE/BigData/'
H = readH(pathH)
print(H.shape)
U = readU(pathU)
Ydata = genNoisy(np.matmul(H, U))
#print('Size of Y:',Y.shape)
beta = 1e5
covY=(1/beta)*np.identity(d)
z_dim=50*seq_length



model = SeqVaeFull()
modelfull = torch.load('output/modelGaussFull1000', map_location={'cuda:0': 'cpu'})
model.load_state_dict(modelfull['state_dict'])
    # ----Loading of previous model ends-----
    # ----- Read H,Y,beta-----





z0 = np.load('Healthy_Z1000.npy')
z0=np.reshape(z0,z_dim,order='F')
c=np.identity(z_dim)
z=MvNormal('z',mu=z0,tau=c)
#z=pymc.Normal('z',mu=1,tau=1)
@pymc.deterministic(plot=False)
def decoderMean(z=z):
    #z=z.reshape(seq_length,1,50)
    Mu, logvar = model.decode(Variable(torch.FloatTensor(z).view(seq_length,1,-1)))  # Converting into torch variable and decoding
    Mu = (Mu.data.view(seq_length, -1).transpose(0,1)).numpy()
    #MuZ = Mu.transpose()

    pathH = '/Users/sg9872/Desktop/Research/Data/Halifax-EC/Simulation/1862/Input/'
    H = readH(pathH)

    MuY=np.matmul(H,Mu)
    return MuY.reshape(seq_length*d,order='F')

@pymc.deterministic(plot=False)
def decoderTau(z=z):
    #z = z.reshape(seq_length, 1, 50)
    Mu, logvar = model.decode(Variable(torch.FloatTensor(z).view(seq_length,1,-1)))  # Converting into torch variable and decoding
    Sigma = logvar.exp()
    Sigma = (Sigma.data.view(seq_length, -1).transpose(0,1)).numpy()
    #SigmaZ = Sigma.transpose()
    SigmaZ=Sigma
    (a,b)=np.shape(SigmaZ)
    TauY=np.eye(d*b)


    for i in range (b):
        vari=SigmaZ[:,i]
        #(g, h) = np.shape(H * vari)
        #print('H size{},H*vari size{}',a,b,g,h)
        covi=np.matmul((H*vari),H.transpose())+(1/beta)*np.eye(d)
        TauY[i*d:(i+1)*d,i*d:(i+1)*d]=np.linalg.inv(covi)
    return TauY

y=MvNormal('y',mu=decoderMean,tau=decoderTau, value=np.reshape(Ydata,d*seq_length,order='F'),observed=True)

print('Reached here! stage 2')

m=pymc.Model([z,y])
mc=pymc.MCMC(m,db='pickle',dbname='pymcTest1')

print('Sampling in process')
mc.sample(iter=250,burn=150)
#pymc.Matplot.plot(mc)

t1=time.time()

print('Time taken for 10 iterations is :',t1-t0)

mc.db.close()

z_sample=mc.trace('z')[:]
print(z_sample)

Mu, logvar = model.decode(Variable(torch.FloatTensor(z_sample).view(seq_length,1,-1)))  # Converting into torch variable and decoding
Sigma = logvar.exp()
Mu = (Mu.data.view(seq_length, -1)).numpy()
Sigma = (Sigma.data.view(seq_length, -1)).numpy()
MuZ = Mu.transpose()
SigmaZ = Sigma.transpose()
    # Sampling ends---------
    #print(MuZ.shape, H.shape)

MeanU, logdetPrecisionU = Posterior(MuZ, SigmaZ, H, Ydata, beta) # Posterior calculation of U given Z, Y
print('Error is',np.linalg.norm(U-MeanU))