from __future__ import print_function

import argparse
import torch
import torch.utils.data
import os
import scipy.io as sio
import numpy as np
from torch import nn, optim
from torch.autograd import Variable
import matplotlib as mtl
import matplotlib.pyplot as plt



modelStructure='SeqGauss'
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

def plotMatrix(M,a,p,q,lbl):
    t = np.arange(a)

    for i in range(p,q):
        y1 = M[:,i:i+1]
        #print('Value of each column is' ,y1)


        line1,=plt.plot( t,y1.numpy(),label=lbl,linewidth=2.5)
    return line1
        #plt.hold('on')
    #plt.hold ('off')
    #plt.xlabel('t')
    #plt.ylabel('Potential')

    #plt.title('TMP curves')
    #plt.show()

if modelStructure=='SeqGauss':
    seq_length = 201
    input_dim = 1898
    
    real_path = os.getcwd()
    os.chdir("../AW1898/ExperimentData/")
    path_data = os.getcwd()
    os.chdir(real_path)
    
    model = SeqVaeFull()
    modelfull=torch.load('output/modelAwFull1200', map_location={'cuda:0': 'cpu'})
    model.load_state_dict(modelfull['state_dict'])


    #-----To test encoder decoder---
    i = 1
    j = 1
    # To import data
    #os.chdir("../DataSingle/")
    #path_data = os.getcwd()
    #os.chdir("../VAE_Pytorch/")
    # data import portion ends
    #TrainData = sio.loadmat(path_data + '/TmpSeg' + str(i) + 'exc' + str(j) + '.mat')

    #U = torch.FloatTensor(TrainData['U'])

    #U = Variable(U.contiguous().view(seq_length, 1, -1))
    #sample,var=model.encode(U)
    #----------
    mean2 =Variable( torch.FloatTensor(np.load(path_data+'/Latent/ZSeg100avg.npy')).view(seq_length,1,-1))
    var2  =Variable( torch.FloatTensor(np.load(path_data+'/Latent/VarSeg100avg.npy')).view(seq_length,1,-1))
    print((var2))
    std=var2.pow(0.5)
    eps=Variable(std.data.new(std.size()).normal_())
    #sample2=eps.mul(std).add_(mean2)
    sample2=(mean2)
    

    for tria in range(2):
        sample1 = Variable(torch.randn(seq_length, 1,50))

        reconstruct1, recVar1 = model.decode(sample1)

        M1=reconstruct1.data.view(seq_length,-1)
        U=M1.numpy()
        sio.savemat('/Users/sg9872/Desktop/Miccai/Miccai18/sampleTMP/NoiseLat'+str(tria)+'.mat', {"U":U})

        line=plotMatrix(M1, seq_length, 161,162,'Line 1')
    first_legend = plt.legend(handles=[line], loc=1)

    # Add the legend manually to the current Axes.
    ax = plt.gca().add_artist(first_legend)
    #mtl.rc('xtick', labelsize=5) 
    #mtl.rc('ytick', labelsize=5)
    plt.xticks(np.array([0,100,200]))
    plt.yticks(np.array([0,0.5,1]))
    ax = plt.gca()
    ax.tick_params(axis = 'both', which = 'major', labelsize = 15)
    #ax.tick_params(axis = 'both', which = 'minor', labelsize = 16)
    
    plt.savefig('/Users/sg9872/Desktop/Miccai/Miccai18/poszsampleTMP.pdf')
    plt.show()
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