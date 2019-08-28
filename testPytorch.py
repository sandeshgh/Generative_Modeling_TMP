from __future__ import print_function

import torch
import torch.utils.data
from torch import nn, optim
from torch.autograd import Variable
import scipy.io as sio
from scipy import linalg as lin
import numpy as np
import math
import matplotlib.pyplot as plt

seq_length = 201
input_dim = 1862

n_samples=1
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
    (M,N)=MuZ.size()
    betaHH=beta*(H.transpose(0,1) @ H)
    PrecisionZ=torch.reciprocal(SigmaZ)
    B=beta*(H.transpose(0,1)@Y)+(PrecisionZ*MuZ)
    #logdetP=0

    for i in range(N):
        PreZ=(PrecisionZ[:,i])
        PrecisionU=betaHH+torch.diag(PreZ)
        Ui,lu=torch.gesv(B[:,i],PrecisionU)
        if i==0:
            MeanU=Ui
        else:
            MeanU=torch.cat((MeanU,Ui),1)
        #logdet=torch.log(torch.det(PrecisionU))
        #logdetP=logdetP+logdet
    return MeanU


def Likelihood_Y_Z(MuZ,SigmaZ,H,Y,beta):
    MuY=H@MuZ
    (d,N)=MuY.size()
    totalTerm=0
    #logdetP=float(0)
    for i in range(N):
        yi=Y[:,i]
        mui=MuY[:,i]
        deviation=yi-mui

        outerProduct=torch.gres(deviation,deviation.transpose(0,1))
        Sigmai=torch.diag(SigmaZ[:,i])
        Covi=(H@Sigmai)@H.transpose(0,1)+(1/beta)*torch.identity(d)
        Pr=torch.inverse(Covi)
        expTerm=torch.sum((outerProduct*Pr))
        logdet=torch.log(torch.det(Pr))
        #print('Determinant is:', determin)
        #print(Covi)
        #prod*=1/(math.exp(0.5*expTerm)*math.sqrt(lin.det(Covi)))
        totalTerm=totalTerm+0.5*(logdet-expTerm)

    return totalTerm

def calculate_loss(Mu,logvar,Upos,H):
    #(d,n)=Y.size()
    #print('Upos is',Upos)
    #print('Mu is',Mu)
    diffSq = (Upos- Mu).pow(2)
    precis = torch.exp(-logvar)
    lossZ = 0.5 * torch.sum(logvar + torch.mul(diffSq, precis))
    lossZ/= (input_dim * seq_length)


    return lossZ

def genNoisy(Y, noisevar=torch.Tensor([0.0001])):
    (a,b)=Y.size()
    noise=noisevar*torch.randn([a,b])
    return Y+noise

#-----Functions below and before main are numpy functions-------------

def readH(path):
    dummy=sio.loadmat(path+'Trans.mat')
    H=dummy['H']
    return H

def readU(path,seg_index=12,exc_index=71):
    TrainData = sio.loadmat(path + 'TmpSeg' + str(seg_index) + 'exc' + str(exc_index) + '.mat')
    U = (TrainData['U'])
    return U.transpose()


def plotMatrix(M,titletext='TMP plot',a=10,b=12):
    (m,n)=M.shape
    t = np.arange(n)
    plt.figure()
    for i in range(a,b):
        y1 = M[i,:]
        #print('Value of each column is' ,y1)


        plt.plot( t,y1)
        #plt.hold('on')
    #plt.hold ('off')
    plt.xlabel('t')
    plt.ylabel('Potential')

    plt.title(titletext)
    plt.show()

def main():
    # ----Load previous learnt model----
    model = SeqVaeFull()
    modelfull = torch.load('output/modelGaussFull400', map_location={'cuda:0': 'cpu'})
    model.load_state_dict(modelfull['state_dict'])
    # ----Loading of previous model ends-----
    # ----- Read H,Y,beta-----
    pathH = '/Users/sg9872/Desktop/Research/Data/Halifax-EC/Simulation/1862/Input/'
    pathU = '/Users/sg9872/Desktop/Research_Projects/Sequence_VAE/BigData/'
    H = torch.FloatTensor(readH(pathH))
    U = torch.FloatTensor(readU(pathU))
    print('Y is :',H @ U)
    Y = genNoisy(H@U)

    beta = torch.FloatTensor([1e5])

    # ------Reading ends------

    for j in range(n_samples):
        print('Sample {} running'.format(j))

        #----Sampling part
        z0 = np.load('Healthy_Z400.npy')
        Z=Variable(torch.FloatTensor(z0).view(seq_length,1,-1), requires_grad=True)
        #Z = Variable(torch.randn([seq_length, 1, 50]), requires_grad=True)
        optimizer = optim.Adam([Z], lr=1e-1)
        Mu, logvar = model.decode(Z)
        for k in range(2):
             #Converting into torch variable and decoding
            #print('Mu is',Mu)
            #print('logvar is ',logvar)
            Sigma=logvar.exp()
            MuZ = (Mu.data.view(seq_length, -1))
            Sigma = (Sigma.data.view(seq_length, -1))
            MuZ=MuZ.transpose(0,1)
            SigmaZ=Sigma.transpose(0,1)
            U_pos=Posterior(MuZ, SigmaZ, H, Y, beta)
            #Sampling ends---------

            #MeanU,logdetPrecisionU=Posterior(MuZ,SigmaZ,H,Y,beta) # Posterior calculation of U given Z, Y

            #logLYZ=Likelihood_Y_Z(MuZ,SigmaZ,H,Y,beta)
            Upos=Variable(U_pos.transpose(0,1).contiguous().view(seq_length,1,-1))
            #for l in range(5):
            optimizer.zero_grad()
            loss = calculate_loss((Mu),(logvar),Upos,Variable(H))
            loss.backward()
            optimizer.step()
            Mu, logvar = model.decode(Z)

            errorU=torch.norm(U_pos-U)
            print('Error in {} iteration is:{}'.format(k,errorU))

        #if j==0 or dataLikelihood> winner_dataLikelihood:
            #winner_dataLikelihood=dataLikelihood
            #winner_index=j
            #winner_U= MeanU

            winner_Z=Z
        # ---Storage ----
        #struct(Z,LYZ,MeanU,dataLikelihood)
   # print('Index of winner',winner_index)
    #print('Likelihood_array', Likelihood_array)
    #print('Error_array', Error_array)
    #np.save('Likelihood_array',Likelihood_array)
    #np.save('Error_array',Error_array)
    plotMatrix(U_pos.numpy(), 'winner', 1406, 1410)
    sio.savemat('Winner_U.mat', {"U":U_pos.numpy()})



if __name__ == "__main__":
    main()