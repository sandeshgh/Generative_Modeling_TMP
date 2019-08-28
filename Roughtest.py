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


x=torch.FloatTensor([[1,2,3],[4,5,6]])
print(x)

y=torch.FloatTensor([[7,8,9],[10,11,12]])
print(y[:,1])
x1=x.contiguous().view(2, 1, -1)
y1=y.contiguous().view(2, 1, -1)
print(x1)

z=torch.cat((x1,y1),1)
print(z[:,1,:])