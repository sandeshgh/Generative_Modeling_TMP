from __future__ import print_function
import argparse
import torch
import torch.utils.data
import os
import scipy.io as sio
from torch import nn, optim
from torch.autograd import Variable

#from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
import numpy as np
import math

# constants defined at the beginning
seq_length=201
input_dim=1898




parser = argparse.ArgumentParser(description='VAE MNIST Example')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=1, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()



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

model = SeqVaeFull()
modelfull = torch.load('output/modelAwFull1200', map_location={'cuda:0': 'cpu'})
model.load_state_dict(modelfull['state_dict'])

real_path = os.getcwd()
os.chdir("../VAE_AW/")

Rec_err=0

seg_range = list(range(1, 16))
seg_range.append(100)

batch_size=len(seg_range)

n_samples=math.floor(input_dim/40)
j = 1
model.eval()
while j < input_dim:
    # Loop over all segments
    for i in seg_range:
        # print('Data/AwTmpSeg' + str(i) + 'exc' + str(j) + '.mat')
        if i == 100:
            nametag = 'AWTmpSeg'
            TrainData = sio.loadmat('Data/' + nametag + str(i) + 'exc' + str(j) + '.mat')
            zz = torch.FloatTensor(TrainData['U'].transpose())
        else:
            nametag = 'AwTmpSeg'
            TrainData = sio.loadmat('Data/' + nametag + str(i) + 'exc' + str(j) + '.mat')
            zz = torch.FloatTensor(TrainData['U'])

        (dim1, dim2) = (zz.shape)
        if dim1 == seq_length:
            U = zz.contiguous().view(seq_length, 1, -1)
        else:
            print('Dimension mismatch')
        # print(U)
        if i == seg_range[0]:
            outData = U
        else:
            outData = torch.cat((outData, U), 1)

    data = Variable(outData)  # sequence length, batch size, input size
    # print(data.size())
    if args.cuda:
        data = data.cuda()

    muTheta, logvarTheta, mu, logvar = model(data)
    recon,rec_var=(model.decode(mu))
    print(mu.size())
    sqErr=torch.sum((recon - data).pow(2))/torch.sum((data).pow(2))
    Rec_err += (1 / batch_size) * sqErr

    #print(pos_z.size())
    j = j + 40
    print(j)
Rec_err=(1/n_samples)*Rec_err
print('Reconstruction error:', Rec_err)
#os.chdir(real_path)
'''
os.chdir("../AW1898/ExperimentData/")
path_data = os.getcwd()
np.save(path_data+'/Latent/ZSeg100avg', pos_z.view(seq_length,-1).numpy())
np.save(path_data+'/Latent/VarSeg100avg', pos_var.view(seq_length,-1).numpy())
os.chdir(real_path)
'''