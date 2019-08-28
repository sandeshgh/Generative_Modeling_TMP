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

n_samples=10
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
    # ------Reading ends------
    LYZ_array=np.zeros(n_samples)
    Likelihood_array = np.zeros(n_samples)
    Error_array = np.zeros(n_samples)
    for j in range(n_samples):
        print('Sample {} running'.format(j))

        #----Sampling part
        Z=np.random.normal(0,1,[seq_length,1,50])
        Mu,logvar=model.decode(Variable(torch.FloatTensor(Z))) #Converting into torch variable and decoding
        Sigma=logvar.exp()
        Mu = (Mu.data.view(seq_length, -1)).numpy()
        Sigma = (Sigma.data.view(seq_length, -1)).numpy()
        MuZ=Mu.transpose()
        SigmaZ=Sigma.transpose()
        #Sampling ends---------

        MeanU,logdetPrecisionU=Posterior(MuZ,SigmaZ,H,Y,beta) # Posterior calculation of U given Z, Y

        logLYZ=Likelihood_Y_Z(MuZ,SigmaZ,H,Y,beta)
        LYZ_array=[LYZ_array,logLYZ]
        dataLikelihood=(logdetPrecisionU)+logLYZ
        errorU=np.linalg.norm(MeanU-U)


        Likelihood_array[j]= dataLikelihood
        Error_array[j]=errorU
        #index=j
        #plotMatrix(MeanU,'Sample'+str(j), 10,15)
        if j==0 or dataLikelihood> winner_dataLikelihood:
            winner_dataLikelihood=dataLikelihood
            winner_index=j
            winner_U= MeanU

            winner_Z=Z
        # ---Storage ----
        #struct(Z,LYZ,MeanU,dataLikelihood)
    print('Index of winner',winner_index)
    print('Likelihood_array', Likelihood_array)
    print('Error_array', Error_array)
    np.save('Likelihood_array',Likelihood_array)
    np.save('Error_array',Error_array)
    np.save('Winner_Z',winner_Z)
    plotMatrix(winner_U, 'winner', 1406, 1410)
    sio.savemat('Winner_U.mat', {"U":winner_U})



if __name__ == "__main__":
    main()