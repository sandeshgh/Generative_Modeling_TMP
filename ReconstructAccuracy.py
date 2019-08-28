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

dataset='AW1898'


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


def plotMatrix(M,a,b):
    t = np.arange(a)
    #print('Vale of time is',t)
    #print('M inside print is',M)
    for i in range(0,b):
        y1 = M[:,i:i+1]
        #print('Value of each column is' ,y1)


        plt.plot( t,y1.numpy())
        #plt.hold('on')
    #plt.hold ('off')
    plt.xlabel('t')
    plt.ylabel('Potential')

    plt.title('TMP curves')
    plt.show()

if modelStructure=='SeqGauss':
    seq_length = 201
    input_dim = 1898

    model1 = SeqVaeFull()
    model2 = SeqVaeFull()
    if dataset=='AW1898':
        j = 441
        modelfull = torch.load('output/modelAwFull600', map_location={'cuda:0': 'cpu'})
        modelfull2 = torch.load('output/modelAwFull1200', map_location={'cuda:0': 'cpu'})
        real_path = os.getcwd()
        os.chdir("../AW1898/ExperimentData/")
        path_data = os.getcwd()
        os.chdir(real_path)

        TrainData = sio.loadmat(path_data + '/Healthy/AWTmpSeg100exc' + str(j) + '.mat')
        Utrue = torch.FloatTensor(TrainData['U'])

        (a,b)=Utrue.size()

        if a==input_dim:
            Utrue = Utrue.transpose(0, 1)

    else:
        j = 101
        modelfull=torch.load('output/modelGaussDeep1000', map_location={'cuda:0': 'cpu'})
        modelfull2 = torch.load('output/modelGaussFull1000', map_location={'cuda:0': 'cpu'})
        real_path = os.getcwd()
        os.chdir("../EC1862/ExperimentData/")
        path_data = os.getcwd()
        os.chdir(real_path)

        TrainData = sio.loadmat(path_data + '/Healthy/TmpSeg100exc' + str(j) + '.mat')
        Utrue = torch.FloatTensor(TrainData['U'])
        Utrue = Utrue.transpose(0, 1)

    model1.load_state_dict(modelfull['state_dict'])



    model2.load_state_dict(modelfull2['state_dict'])
    #-----To test encoder decoder---
    i = 12

    # To import data

    (a, b) = Utrue.size()
    if a==seq_length:
        U = Variable(Utrue.contiguous().view(seq_length, 1, -1))
    else:
        print('Size mismatch of U',a)
        #return None

    sample1,var1=model1.encode(U)
    #np.save(path_data+'Latent/ZSeg100exc'+str(j), sample1.data.view(seq_length,-1).numpy())

    sample2,var2=model2.encode(U)
    np.save(path_data+'/Latent/ZSeg100exc'+str(j), sample2.data.view(seq_length,-1).numpy())
    #----------


    #sample = Variable(torch.randn(seq_length, 1,50))
    sample2 =Variable( torch.FloatTensor(np.load(path_data+'/Latent/ZSeg100avg.npy')).view(seq_length,1,-1))
    #sample2=Variable((modelfull2['pos_z']).view(seq_length,1,-1))
    print(sample2.size())

    reconstruct1, recVar1 = model1.decode(sample1)
    reconstruct2, recVar2 = model2.decode(sample2)
    M1=reconstruct1.data.view(seq_length,-1)
    M2=reconstruct2.data.view(seq_length,-1)

    Error1=torch.norm(M1-Utrue)
    Error2=torch.norm(M2-Utrue)
    print('Error m100:{}, Error m1000:{}'.format(Error1,Error2))
    plotMatrix(M2, seq_length, 5)
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
#print('M is',M)
#print('V is ',V)
#plotMatrix(V, seq_length,2)