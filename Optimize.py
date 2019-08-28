import torch
import torch.utils.data
from torch import nn, optim
from torch.autograd import Variable
import scipy.io as sio
from scipy import linalg as lin
import numpy as np
import math

import pybobyqa
from skopt import forest_minimize

seq_length=201
input_dim=1862

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

class VAEGauss(nn.Module):
    def __init__(self):
        super(VAEGauss, self).__init__()

        self.fc1 = nn.LSTM(input_dim, 800)
        self.fc21 = nn.Linear(800, 50)
        self.fc22 = nn.Linear(800, 50)
        self.fc3 = nn.Linear(50, 800)
        self.fc41 = nn.LSTM(800, input_dim)
        self.fc42 = nn.LSTM(800, input_dim)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()


    def encode(self, x):
        out, hidden=self.fc1(x)
        h1 = self.relu(out)
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z):
        h3 = self.relu(self.fc3(z))
        out1,hidden1=self.fc41(h3)
        out2, hidden2 = self.fc42(h3)
        return self.sigmoid(out1), (out2)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        muTheta,logvarTheta=self.decode(z)
        return muTheta,logvarTheta, mu, logvar


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


def Likelihood_Y_Z(MuZ,SigmaZ,H,Y,beta):
    MuY=np.matmul(H,MuZ)
    (d,N)=MuY.shape
    totalTerm=0
    #logdetP=float(0)
    for i in range(N):
        yi=Y[:,i]
        mui=MuY[:,i]
        deviation=yi-mui
        deviation.shape=(d,1)
        #print('deviation is ',deviation)
        outerProduct=deviation.dot(deviation.transpose())
        Sigmai=np.diag(SigmaZ[:,i])
        Covi=np.matmul(np.matmul(H,Sigmai),H.transpose())+(1/beta)*np.identity(d)
        Pr=np.linalg.inv(Covi)
        expTerm=np.sum(np.multiply(outerProduct,Pr))
        (s,logdet)=np.linalg.slogdet(Pr)
        #print('Determinant is:', determin)
        #print(Covi)
        #prod*=1/(math.exp(0.5*expTerm)*math.sqrt(lin.det(Covi)))
        totalTerm=totalTerm+0.5*(logdet-expTerm)

    return totalTerm
def genNoisy(Y,noisevar=1e-4,index=2):
    (a,b)=Y.shape
    noise=np.random.normal(0, math.sqrt(noisevar), [a,b])
    return Y+noise

def readH(path):
    dummy=sio.loadmat(path+'Trans.mat')
    H=dummy['H']
    return H

def readU(path,seg_index=12,exc_index=71):
    TrainData = sio.loadmat(path + 'TmpSeg' + str(seg_index) + 'exc' + str(exc_index) + '.mat')
    U = (TrainData['U'])
    return U.transpose()

def optimizing_func(Z):
    z0 = np.load('Healthy_Z1000.npy')
    z0 = z0.reshape((50 * seq_length))
    lossZ=0.5*0.1*np.sum(np.power(Z-z0,2))
    Z=Z.reshape(seq_length,1,-1)
    model = SeqVaeFull()
    modelfull = torch.load('output/modelGaussFull1000', map_location={'cuda:0': 'cpu'})
    model.load_state_dict(modelfull['state_dict'])
    # ----Loading of previous model ends-----
    # ----- Read H,Y,beta-----
    pathH = '/Users/sg9872/Desktop/Research/Data/Halifax-EC/Simulation/1862/Input/'
    pathU = '/Users/sg9872/Desktop/Research_Projects/Sequence_VAE/BigData/'
    H = readH(pathH)
    U = readU(pathU)
    Y = genNoisy(np.matmul(H, U))
    beta = 1e5

    Mu, logvar = model.decode(Variable(torch.FloatTensor(Z)))  # Converting into torch variable and decoding
    Sigma = logvar.exp()
    Mu = (Mu.data.view(seq_length, -1)).numpy()
    Sigma = (Sigma.data.view(seq_length, -1)).numpy()
    MuZ = Mu.transpose()
    SigmaZ = Sigma.transpose()
    # Sampling ends---------
    print(MuZ.shape)

    #MeanU, logdetPrecisionU = Posterior(MuZ, SigmaZ, H, Y, beta)  # Posterior calculation of U given Z, Y

    logLYZ = Likelihood_Y_Z(MuZ, SigmaZ, H, Y, beta)
    return -(logLYZ)+lossZ

def main():
    lowerbound=-3*np.ones(50*seq_length)
    upperbound=3*np.ones(50*seq_length)
    z0=np.load('Healthy_Z1000.npy')
    print('Z0 is',z0)
    z0=z0.reshape((50*seq_length))
    print('Z0 is', z0)
    soln=pybobyqa.solve(optimizing_func,z0,maxfun=100)
    print(soln)
    print('X part is',soln.x)

    Z = soln.x
    #Z=z0
    Z = Z.reshape(seq_length, 1, -1)
    model = SeqVaeFull()
    modelfull = torch.load('output/modelGaussFull1000', map_location={'cuda:0': 'cpu'})
    model.load_state_dict(modelfull['state_dict'])
    # ----Loading of previous model ends-----
    # ----- Read H,Y,beta-----
    pathH = '/Users/sg9872/Desktop/Research/Data/Halifax-EC/Simulation/1862/Input/'
    pathU = '/Users/sg9872/Desktop/Research_Projects/Sequence_VAE/BigData/'
    H = readH(pathH)
    U = readU(pathU)
    Y = genNoisy(np.matmul(H, U))
    beta = 1e5

    Mu, logvar = model.decode(Variable(torch.FloatTensor(Z)))  # Converting into torch variable and decoding
    Sigma = logvar.exp()
    Mu = (Mu.data.view(seq_length, -1)).numpy()
    Sigma = (Sigma.data.view(seq_length, -1)).numpy()
    MuZ = Mu.transpose()
    SigmaZ = Sigma.transpose()
    # Sampling ends---------
    #print(MuZ.shape, H.shape)

    MeanU, logdetPrecisionU = Posterior(MuZ, SigmaZ, H, Y, beta) # Posterior calculation of U given Z, Y
    print('Error is',np.linalg.norm(U-MeanU))
    sio.savemat('Useg12exc71solBob.mat', {"U": MeanU})

if __name__ == "__main__":
    main()