from __future__ import print_function

import argparse
import torch
import torch.utils.data
import os
import scipy.io as sio
import numpy as np
from torch import nn, optim
from torch.autograd import Variable
import matplotlib.pyplot as plt



modelStructure='SeqGauss'

class VAEBernoulli(nn.Module):
    def __init__(self):
        super(VAEBernoulli, self).__init__()

        self.fc1 = nn.LSTM(input_dim, 400)
        self.fc21 = nn.Linear(400, 20)
        self.fc22 = nn.Linear(400, 20)
        self.fc3 = nn.Linear(20, 400)
        self.fc4 = nn.LSTM(400, input_dim)

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
        out,hidden=self.fc4(h3)
        return self.sigmoid(out)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

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

class VECBern(nn.Module):
    def __init__(self):
        super(VECBern, self).__init__()

        self.fc1 = nn.Linear(input_dim, 800)
        self.fc21 = nn.Linear(800, 50)
        self.fc22 = nn.Linear(800, 50)
        self.fc3 = nn.Linear(50, 800)
        self.fc4 = nn.Linear(800, input_dim)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def encode(self, x):
        h1 = self.relu(self.fc1(x))
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
        return self.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

def plotMatrix(M,a,b):
    t = np.arange(a)
    print('Vale of time is',t)
    print('M inside print is',M)
    #for i in range(0,b):
    y1 = M[:,0]
    print('y1 is',y1)
        #print('Value of each column is' ,y1)


    plt.plot( t,y1.data.numpy())


    plt.xlabel('t')
    plt.ylabel('Potential')

    plt.title('TMP curves')
    plt.show()

if modelStructure=='SeqGauss':
    seq_length = 201
    input_dim = 1862
    i=1
    j=1
    # To import data
    os.chdir("../DataSingle/")
    path_data = os.getcwd()
    os.chdir("../VAE_Pytorch/")
    # data import portion ends
    TrainData = sio.loadmat(path_data + '/TmpSeg' + str(i) + 'exc' + str(j) + '.mat')
    V=TrainData['U']
    print('V is ', V)

    U = Variable(torch.FloatTensor(TrainData['U']))
    M = U

    print('U is ', U)
    #U=U.contiguous
    U = U.contiguous().view(seq_length, 1, -1)
    #U=Variable(U.permute(0, 2, 1))

    model = VAEGauss()
    modelfull=torch.load('output/modelGaussAn100', map_location={'cuda:0': 'cpu'})
    model.load_state_dict(modelfull['state_dict'])
    #model=modelfull['state_dict']
    #print(model)
    #mu, logvar = model.encode(U)
    model.eval()

    #sample=mu
    #sample = Variable(torch.randn(seq_length, 1,50))

    #reconstruct, recVar = model.decode(sample)
    #reconstruct=U
    #print('The reconstruction is')
    #print(reconstruct)
    #M=reconstruct.data.resize_(seq_length,input_dim)
    #M=V.resize_(seq_length,input_dim)
    #print('V is ',V)
    #print('M is', M)

    plotMatrix(M, seq_length, 2)
elif modelStructure=='VecBern':
    input_dim = 3724
    model = VECBern()
    modelfull = torch.load('output/model100', map_location={'cuda:0': 'cpu'})
    model.load_state_dict(modelfull['state_dict'])
    # model=modelfull['state_dict']
    # print(model)
    model.eval()

    sample = Variable(torch.randn( 1, 50))

    reconstruct= model.decode(sample)
    print('The reconstruction is')
    print(reconstruct)
    M = reconstruct.data.resize_(input_dim,1)
    plt.plot(np.arange(input_dim),M.numpy(),'r^')
    plt.show()
