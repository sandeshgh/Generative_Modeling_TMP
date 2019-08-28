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

# constants defined at the beginning
seq_length=201
input_dim=1862
batch_size=17
n_samples=37
isAnnealing=1


parser = argparse.ArgumentParser(description='VAE MNIST Example')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()



torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=True, **kwargs)

#To import data
os.chdir("../DataSingle/")
path_data=os.getcwd()
os.chdir("../VAE_Pytorch/")
#data import portion ends

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

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


model = VAE()
if args.cuda:
    model.cuda()


def loss_function(muTheta,logvarTheta, x, mu, logvar,annealParam):
    #tol=1e-8
    #BCE = -torch.sum(torch.mul(x,torch.log(tol+recon_x))+(1-x).mul(torch.log(1+tol-recon_x)))
    diffSq=(x-muTheta).pow(2)
    precis=torch.exp(-logvarTheta)
    #print('Sum logvar:',torch.sum(logvarTheta))
    #print('Sumerror: ',torch.sum(diffSq))
    #print('SumerrordivVar: ', torch.sum(torch.mul(diffSq,precis)))
    BCE=0.5*torch.sum(logvarTheta+torch.mul(diffSq,precis))
    BCE/=(batch_size * input_dim*seq_length)

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * annealParam*torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    # Normalise by same number of elements as in reconstruction
    KLD /= batch_size * 50*seq_length

    return BCE + KLD


optimizer = optim.Adam(model.parameters(), lr=1e-3)

def prepareData(train_loader, seq_length, batch_index):
    index_start=batch_index*seq_length
    index_end=index_start+seq_length
    for i, (data, _) in enumerate(train_loader):
        data=data.view(-1,input_dim)
        data=data.resize_((batch_size,input_dim,1))
        if i==index_start:
            outData=data
        elif i > index_start and i < index_end:
            outData=torch.cat((outData,data),2)
    return outData.permute(2,0,1)
def train(epoch):
    model.train()
    train_loss = 0
    seg_range = list(range(0, 17))
    j = 1
    while j < input_dim:
        # Loop over all segments
        for i in seg_range:
            # batch_xs, _ = mnist.train.next_batch(batch_size)

            # print (i,',',j)
            TrainData = sio.loadmat(path_data + '/TmpSeg' + str(i) + 'exc' + str(j) + '.mat')

            zz = torch.FloatTensor(TrainData['U'])
            U=zz.contiguous().view(seq_length, 1, -1)
            #print(U)
            if i == 0:
                outData = U
            elif i > 0 and i < 17:
                outData = torch.cat((outData, U), 1)

        data = Variable(outData)  #sequence length, batch size, input size
        #print(data.size())
        if args.cuda:
            data = data.cuda()
        optimizer.zero_grad()
        muTheta,logvarTheta, mu, logvar = model(data)
        if epoch<50:
            annealParam=0
        elif epoch <500:
            annealParam=(epoch/500)
        else:
            annealParam=1
        loss = loss_function(muTheta,logvarTheta, data, mu, logvar,annealParam)
        loss.backward()
        train_loss += loss.data[0]
        optimizer.step()

        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, j, 50*n_samples,
            100. * j / (50*n_samples),
            loss.data[0] ))
        j=j+50

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / n_samples))


def test(epoch):
    model.eval()
    test_loss = 0
    for i, (data, _) in enumerate(test_loader):
        if args.cuda:
            data = data.cuda()
        data = Variable(data, volatile=True)
        recon_batch, mu, logvar = model(data)
        test_loss += loss_function(recon_batch, data, mu, logvar).data[0]
        if i == 0:
            n = min(data.size(0), 8)
            comparison = torch.cat([data[:n],
                                  recon_batch.view(args.batch_size, 1, 28, 28)[:n]])
            save_image(comparison.data.cpu(),
                     'results/reconstruction_' + str(epoch) + '.png', nrow=n)

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))

def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)

for epoch in range(1, args.epochs + 1):
    train(epoch)
    if (epoch % 100==0):
        save_checkpoint({
            'epoch': epoch ,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, 'output/modelGaussAn' + str(epoch))
    '''
    test(epoch)
    sample = Variable(torch.randn(64, 20))
    if args.cuda:
        sample = sample.cuda()
    sample = model.decode(sample).cpu()
    save_image(sample.data.view(64, 1, 28, 28),
               'results/sample_' + str(epoch) + '.png')
    '''
