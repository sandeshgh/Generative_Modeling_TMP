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

dataset='EC1862'
input_dim = 1862

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


class SeqVAEsmall(nn.Module):
    def __init__(self):
        super(SeqVAEsmall, self).__init__()

        self.fc1 = nn.LSTM(input_dim, 500)
        self.fc21 = nn.LSTM(500, 10)
        self.fc22 = nn.LSTM(500, 10)
        self.fc3 = nn.LSTM(10, 500)
        self.fc41 = nn.LSTM(500, input_dim)
        self.fc42 = nn.LSTM(500, input_dim)
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

    i = 5

    model1 = SeqVaeFull()
    model2 = SeqVaeFull()
    if dataset=='AW1898':
        j = 81
        input_dim = 1898
        modelfull = torch.load('output/modelAwFull600', map_location={'cuda:0': 'cpu'})
        modelfull2 = torch.load('output/modelAwFull1200', map_location={'cuda:0': 'cpu'})
        real_path = os.getcwd()
        os.chdir("../AW1898/BigData")
        path_data = os.getcwd()
        os.chdir(real_path)
        path_out="/Users/sg9872/Desktop/Research_Projects/Sequence_VAE/AnalyzeVisual/"
        fileName='Seg'+str(i)+'exc' + str(j)
        TrainData = sio.loadmat(path_data + '/AwTmp'+fileName+ '.mat')
        Utrue = torch.FloatTensor(TrainData['U'])

        (a,b)=Utrue.size()

        if a==input_dim:
            Utrue = Utrue.transpose(0, 1)

    elif dataset=='EC1862':
        j = 101

        modelfull = torch.load('output/modelGaussFull400', map_location={'cuda:0': 'cpu'})
        modelfull2 = torch.load('output/modelGaussFull1000', map_location={'cuda:0': 'cpu'})
        real_path = os.getcwd()
        os.chdir("../EC1862/BigData")
        path_data = os.getcwd()
        os.chdir(real_path)
        path_out="/Users/sg9872/Desktop/Research_Projects/Sequence_VAE/AnalyzeVisual/EC1862/"
        fileName='Seg'+str(i)+'exc' + str(j)
        TrainData = sio.loadmat(path_data + '/Tmp'+fileName+ '.mat')
        Utrue = torch.FloatTensor(TrainData['U'])

        fileName12 = 'Seg' + str(12) + 'exc' + str(j)
        TrainData12 = sio.loadmat(path_data + '/Tmp' + fileName12 + '.mat')
        Utrue12 = torch.FloatTensor(TrainData12['U'])


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


    # To import data

    (a, b) = Utrue.size()
    if a==seq_length:
        U = Variable(Utrue.contiguous().view(seq_length, 1, -1))
        U12= Variable(Utrue12.contiguous().view(seq_length, 1, -1))
    else:
        print('Size mismatch of U',a)
        #return None

    sample1,var1=model1.encode(U)
    #np.save(path_data+'Latent/ZSeg100exc'+str(j), sample1.data.view(seq_length,-1).numpy())

    sampleU12, var2 = model2.encode(U12)
    Z12 = sampleU12.data.view(seq_length, -1).numpy()


    sample2,var2=model2.encode(U)
    Z=sample2.data.view(seq_length,-1).numpy()
    VarianceMat=var2.data.view(seq_length,-1).numpy()
    #np.save(path_out+'ZSeg100exc'+str(j), sample2.data.view(seq_length,-1).numpy())
    sio.savemat(path_out+'Z'+fileName+ '.mat', {"Z": Z})
    sio.savemat(path_out + 'Var'+fileName + '.mat', {"V": VarianceMat})
    reconstruct2, recVar2 = model2.decode(sample2)
    R2 = reconstruct2.data.view(seq_length, -1).numpy()
    sio.savemat(path_out + 'Rec'+fileName + '.mat', {"R": R2})
    #----------


    #sample = Variable(torch.randn(seq_length, 1,50))
    #sample2 =Variable( torch.FloatTensor(np.load(path_data+'/Latent/ZSeg100avg.npy')).view(seq_length,1,-1))
    #sample2=Variable((modelfull2['pos_z']).view(seq_length,1,-1))
    print(sample2.size())
    '''
    for alpha in range(3,4):
        sampleNew=Z;
        sampleNew[:,(range(1,25))]=(alpha-3)*0.25*Z[:,(range(1,25))]
        sio.savemat(path_out + 'Z'+ fileName + '_1_25alpha'+str(alpha)+'.mat', {"Z": sampleNew})
        sampleN = Variable(torch.FloatTensor(sampleNew).view(seq_length, 1, -1))
        reconstruct2, recVar2 = model2.decode(sampleN)
        R2 = reconstruct2.data.view(seq_length, -1).numpy()
        sio.savemat(path_out + 'Rec'+ fileName + '_1_25alpha'+str(alpha)+'.mat', {"R": R2})
    '''
    for beta in range(11):
        sampleNew=Z;
        sampleNew[(range(50,100)),:]=(beta/10)*Z12[(range(50,100)),:]+(1-beta/10)*Z[(range(50,100)),:]
        sio.savemat(path_out + 'Z'+ fileName + '_50_100beta'+str(beta)+'.mat', {"Z": sampleNew})
        sampleN = Variable(torch.FloatTensor(sampleNew).view(seq_length, 1, -1))
        reconstruct2, recVar2 = model2.decode(sampleN)
        R2 = reconstruct2.data.view(seq_length, -1).numpy()
        sio.savemat(path_out + 'Rec'+ fileName + '_50_100beta'+str(beta)+'.mat', {"R": R2})

    reconstruct1, recVar1 = model1.decode(sample1)

    M1=reconstruct1.data.view(seq_length,-1)
    #

    Error1=torch.norm(M1-Utrue)
    Error2=torch.norm(R2-Utrue)
    print('Error m100:{}, Error m1000:{}'.format(Error1,Error2))
    #plotMatrix(M2, seq_length, 5)
