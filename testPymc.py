from __future__ import print_function

import torch
import torch.utils.data
from torch import nn, optim
from torch.autograd import Variable
#import theano
#import theano.tensor as tt

import numpy as np
import math
import pymc
from pymc import MvNormal
from pymc import deterministic
import scipy.io as sio

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
z0=z0.reshape(z_dim)
c=np.identity(z_dim)
z=MvNormal('z',mu=z0,tau=c)
#z=pymc.Normal('z',mu=1,tau=1)
@pymc.deterministic(plot=False)
def decoderMean(z=z):
    z=z.reshape(seq_length,1,50)
    Mu, logvar = model.decode(Variable(torch.FloatTensor(z)))  # Converting into torch variable and decoding
    Mu = (Mu.data.view(seq_length, -1)).numpy()
    MuZ = Mu.transpose().reshape(input_dim*seq_length)
    return MuZ

@pymc.deterministic(plot=False)
def decoderTau(z=z):
    z = z.reshape(seq_length, 1, 50)
    Mu, logvar = model.decode(Variable(torch.FloatTensor(z)))  # Converting into torch variable and decoding
    Sigma = logvar.exp()
    Sigma = (Sigma.data.view(seq_length, -1)).numpy()
    SigmaZ = Sigma.transpose().reshape(input_dim*seq_length)
    Tau=np.reciprocal(SigmaZ)
    return np.diag(Tau)

x=MvNormal('x',mu=decoderMean,tau=decoderTau)

@deterministic(plot=False)
def meanY(x=x):
    pathH = '/Users/sg9872/Desktop/Research/Data/Halifax-EC/Simulation/1862/Input/'
    H = readH(pathH)
    (a,b)=H.shape
    Hkron=np.kron(np.eye(b),H)
    return np.matmul(Hkron,x)

y = MvNormal('y', mu=meanY, tau=beta, value=Ydata.reshape(d*seq_length),observed=True)
m=pymc.Model([z,x,y])
mc=pymc.MCMC(m)
mc.sample(iter=10,burn=5)
pymc.Matplot.plot(mc)
