import math
import random
import unittest
import itertools
import contextlib
from copy import deepcopy
from itertools import repeat, product
from functools import wraps, reduce
from operator import mul
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import torch as t
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel as dp
import torch.nn.init as init
import torch.nn.utils as nn_utils
import torch.nn.utils.rnn as rnn_utils
from torch.nn.utils import clip_grad_norm
from torch.autograd import Variable, gradcheck
from torch.nn import Parameter



'''
NT max-time-steps in a sequence
NB sequences, each of dim NF (i.e. num feature)

CAUTION: we don't support NF > 1, so just keep it at 1
'''
def make_diff_data(NT, NB, NF=1, delay=0, test = 0.3, gpu = False):

    #NT, NB, NF, delay = 15, 1000, 1, 3
    dtype = t.FloatTensor
    # x is a random binary tensor
    x = t.Tensor(np.random.random_integers(0,1,NT*NB)).view(NT,NB,NF).type(dtype)

    # lengths of the sequences
    # (each batch is just a 1-D sequence)


    y = t.zeros(x.size())

    # fill in each seq in y as the "diff" of the seq in x (floored by 0, so no negatives)
    # In each seq, we fill in a 0 at the start, to make the x, y tensors same shape

    y[2+delay:, :, :] = t.max(t.zeros(x.size()), x - t.cat( [t.zeros(1,NB,NF), x[:-1, :, :]]))[2:-delay, :, :]

    # check one pair of (x, y) sequences:
    # y is basically diff(x), with a 0 added on at the beginning,
    # i.e. y(t) = max(0, x(t) - x(t-1))
    # x[:, 0, :].numpy().flatten()
    # y[:, 0, :].numpy().flatten()
    if gpu:
        var_y = Variable(t.Tensor(y).type(dtype).cuda())
        var_x = Variable(x.cuda())
    else:
        var_y = Variable(t.Tensor(y).type(dtype))
        var_x = Variable(x)


    nb_test = int(test * NB)
    nb_train = NB - nb_test
    train_lengths = -np.sort(-np.random.random_integers(NT / 2, NT, nb_train))
    test_lengths = -np.sort(-np.random.random_integers(NT / 2, NT, nb_test))

    x_train = var_x[:, :nb_train, :]
    x_test = var_x[:, :nb_test, :]
    y_train = var_y[:, :nb_train, :]
    y_test = var_y[:, :nb_test, :]

    # var_x.cuda() # on aws/gpu
    # packed_var_x = rnn_utils.pack_padded_sequence(var_x, lengths, batch_first=False)
    return x_train, y_train, train_lengths, x_test, y_test, test_lengths

def repackage_hidden(h):
    """Wraps hidden states in new Variables, to detach them from their history."""
    if type(h) == Variable:
        return Variable(h.data)
    else:
        return tuple(repackage_hidden(v) for v in h)

class RNN(nn.Module):
    def __init__(self, rnn_type = 'GRU', dropout = 0.7, nf = 1, nh = 5, nlay = 2):
        super(RNN, self).__init__()
        self.nf = nf # num input features
        self.nh = nh # num hidden units (hidden layer size)
        self.nlay = nlay # num recurrent layers
        self.rnn_type = rnn_type  # 'LSTM' or 'GRU'
        self.drop = nn.Dropout(dropout)
        self.rnn = getattr(nn, rnn_type)(nf, nh, nlay, dropout=dropout)
        self.fc =  nn.Linear(nh, 1)
        self.sig = nn.Sigmoid()

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        if self.rnn_type == 'LSTM':
            return (Variable(weight.new(self.nlay, bsz, self.nh).zero_()),
                    Variable(weight.new(self.nlay, bsz, self.nh).zero_()))
        else:
            return Variable(weight.new(self.nlay, bsz, self.nh).zero_())

    def forward(self,x, hid):
        out, hid = self.rnn(x, hid)
        isPacked = (type(out) != Variable)
        if isPacked:
            out = out.data
        out = self.drop(out)
        if not isPacked:
            out = out.view(-1, out.size(2))
        h = self.fc(out)
        pred = self.sig(h)
        return pred, hid

# best settings:
#model = RNN(rnn_type ='GRU', nh=20, nlay=2, dropout = 0.5)
model = RNN(rnn_type ='GRU', nh=10, nlay=2, dropout = 0.0)
#loss_fn = nn.MSELoss(size_average=True)
loss_fn = nn.BCELoss(size_average=True)
optimizer = optim.Adam(model.parameters(), lr = 5e-4)
#optimizer = optim.Adam(model.parameters(), lr = 0.7)
#optimizer = optim.RMSprop(model.parameters(), lr = 0.001)
model.zero_grad()

var_x, var_y, lengths, xtest, ytest, test_lens = make_diff_data(15, 1000, 1, 3)
## manually run model a few iterations


bsiz = 2 # mini-batches of 50 sequences at a time
nb = var_x.data.size()[1]/bsiz  # how many mini-batches per epoch

for epoch in tqdm(range(100)):
    for batch in range(nb):
        bStart = batch * bsiz
        bEnd = min(var_x.size()[1], (batch + 1)*bsiz)
        hidden = model.init_hidden(bsiz)
        hidden = repackage_hidden(hidden)
        model.zero_grad()
        batch_x = var_x[:, bStart : bEnd, :]
        batch_lengths = lengths[bStart : bEnd]
        batch_x_packed = rnn_utils.pack_padded_sequence(batch_x, batch_lengths)
        y_pred, hidden = model(batch_x_packed, hidden)
        batch_y = rnn_utils.pack_padded_sequence(var_y[:, bStart : bEnd, :], batch_lengths)
        # CAUTION -- y_pred is BATCH-WISE, i.e. b
        # batch 0 for ALL sequences, then
        # batch 2 for ALL sequences, etc.
        # target = var_y.squeeze()[:, bStart:bEnd].contiguous().view(-1, 1)
        loss = loss_fn(y_pred, batch_y.data)
        # optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print 'loss = ', loss.data[0]
    ny = min(bsiz,15)
    #sys.stdout.write('\ry,pred = ' + str(zip(batch_y.data.squeeze()[:ny].data.numpy(), y_pred.data.squeeze()[:ny].numpy())) + '\r')
    # print model(x_test_packed).data.numpy().flatten()
    # print params[0].grad.data.numpy().flatten()
    # print params[0].data.numpy().flatten()


### examine params

# validation: test out of sample
yp, _ = model (xtest, None)
loss_fn(yp, ytest)

# check prediction vs true
zip(yp.squeeze().data.numpy(), ytest.contiguous().view(-1,1).squeeze().data.numpy())[-15:]

