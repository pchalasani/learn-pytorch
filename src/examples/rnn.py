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
from sklearn.preprocessing import OneHotEncoder
from numpy.lib.stride_tricks import as_strided

# enc = OneHotEncoder()
# enc.fit(np.array([0,3,9]).reshape(-1,1))
# enc.transform(np.array([0,3,9,0,9,3,0,0]).reshape(-1,1)).toarray()
#
def split_train_test(x,y, test = 0.3, minlen = 2):
    nt, nb, nf = x.size()
    nb_test = int(test * nb)
    nb_train = nb - nb_test
    train_lengths = -np.sort(-np.random.random_integers(nt/minlen, nt, nb_train))
    test_lengths = -np.sort(-np.random.random_integers(nt/minlen, nt, nb_test))

    x_train = x[:, :nb_train, :]
    x_test = x[:, nb_train:, :]
    y_train = y[:, :nb_train, :]
    y_test = y[:, nb_train:, :]

    return x_train, y_train, train_lengths, x_test, y_test, test_lengths


def one_hot(x):
    """
    convert a tensor x (nt, nb, 1) => one-hot encode the third dimension, i.e.
    to a tensor (nt, nb, nf) where nf = num unique values of tensor elements
    """
    enc = OneHotEncoder()
    if type(x) == t.Tensor:
        vals = x.contiguous().view(-1,1).numpy()
        enc.fit(vals)
        return t.Tensor(enc.transform(vals).toarray()).view(x.size()[0],x.size()[1],-1)
    else:
        vals = np.transpose(np.array([x]))
        enc.fit(vals)
        return enc.transform(vals).toarray()

def genx(size):
    k = 6 # should be > 2
    x = np.random.choice(2, k)
    for i in range(size-k):
        n1 = sum(x[i-k : i-2])
        if n1 > k/2:
            thresh = 0.1
        else:
            thresh = 0.9
        x = np.append(x, 1 * np.random.rand() < thresh)
    return x


def geny(x, i):
    '''
    y[i] as a function of the entire x, which is typically a window of x values
    :param x:
    :param i:
    :return:
    '''
    return 1 * (sum(x[i:]) > 0)



def make_xy_seq_data(genx, geny, xstart, xend=0, nt = 15, nb = 50, nf = 1, test = 0.3, fill = 'rand', minlen = 0.5, gpu = False, xhot = False, yhot = False):
    '''
    :param genx: x seq generator
    :param geny: y seq generator
    :param xstart:  y[i] is a function of x[i-xstart : i+xend]
    :param xend:
    :param nt: (max) num time steps
    :param nb: num batches
    :param nf: num features
    :param test:
    :param fill:
    :param minlen:
    :param gpu:
    :param xhot:
    :param yhot:
    :return:
    '''
    size = nt * nb * nf
    X = genx(size)
    Y = []
    for i in range(size): # don't care about indexes here
        Y = np.append(Y, geny(X[i-xstart : i + xend], xstart))

    tx = t.Tensor(X).view(nb,nt,nf).transpose(0,1)
    ty = t.Tensor(Y).view(nb,nt,nf).transpose(0,1)
    # fill in initial "xstart" elements of each seq with rand values
    if fill == 'zero':
        ty[:xstart, :, :] = 0.0
    if xhot:
        tx = one_hot(tx)
    if yhot:
        ty = one_hot(ty)


    if gpu:
        x = tx.cuda()
        y = ty.cuda()
    else:
        x = tx
        y = ty

    lengths = -np.sort(-np.random.random_integers( int(nt * minlen), nt, nb))
    return x, y, lengths

#x, y, lens = make_xy_seq_data(genx, geny, 6, 3, nt = 15, nb = 5)

# generate batches of sequential binary data
# defined by an arbitrary rule
def make_seq_data(genfn, delay, nt = 10, nb = 50, nf = 1, fill = 'rand', minlen = 0.5, rand= True, gpu = False, xhot = False, yhot = False):
    size = nt * nb * nf
    if rand:
        # add extra elems at start so we get all legit values for Y
        # This could be important as it affects the distribution of Y
        X = np.array(np.random.choice(2, size=(size,)))
    else: # just for testing
        X = np.array(range(size))
    Y = []
    for i in range(size):
        Y = np.append(Y, genfn(X[i-delay:i]))

    tx = t.Tensor(X).view(nb,nt,nf).transpose(0,1)
    ty = t.Tensor(Y).view(nb,nt,nf).transpose(0,1)
    # fill in initial "delay" elements of each seq with rand values
    if fill == 'zero':
        ty[:delay, :, :] = 0.0
    if xhot:
        tx = one_hot(tx)
    if yhot:
        ty = one_hot(ty)


    if gpu:
        x = tx.cuda()
        y = ty.cuda()
    else:
        x = tx
        y = ty
    lengths = -np.sort(-np.random.random_integers( int(nt * minlen), nt, nb))
    return x, y, lengths




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

    return split_train_test(var_x, var_y, test)

def make_r2rt_data(nt = 10, nb = 50, nf = 1, test = 0.3, gpu = False, xhot = False, yhot = False):
    size = nt * nb * nf
    X = np.array(np.random.choice(2, size=(size,)))
    Y = []
    for i in range(size):
        threshold = 0.5
        if X[i-3] == 1:
            threshold += 0.5
        if X[i-8] == 1:
            threshold -= 0.25
        if np.random.rand() > threshold:
            Y.append(0)
        else:
            Y.append(1)

    tx = t.Tensor(X).view(nb,nt,nf).transpose(0,1)
    ty = t.Tensor(Y).view(nb,nt,nf).transpose(0,1)
    if xhot:
        tx = one_hot(tx)
    if yhot:
        ty = one_hot(ty)

    if gpu:
        x = Variable(tx.cuda())
        y = Variable(ty.cuda())
    else:
        x = Variable(tx)
        y = Variable(ty)

    return split_train_test(x,y,test)

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
        # NOTE: we're assuming num output-features is 1 regardless of num input-features
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
