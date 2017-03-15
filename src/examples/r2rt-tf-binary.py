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
import numpy as np
# import tensorflow as tf
# %matplotlib inline
import matplotlib.pyplot as plt
from rnn import *

num_steps = 5 # number of truncated backprop steps ('n' in the discussion above)
batch_size = 200
num_classes = 2
state_size = 4
learning_rate = 0.1



# best settings:
#model = RNN(rnn_type ='GRU', nh=20, nlay=2, dropout = 0.5)
model = RNN(rnn_type ='GRU', nh=50, nlay=32, dropout = 0.7)
#loss_fn = nn.MSELoss(size_average=True)
loss_fn = nn.BCELoss(size_average=True)
optimizer = optim.Adam(model.parameters(), lr = 5e-4)
#optimizer = optim.Adam(model.parameters(), lr = 0.7)
#optimizer = optim.RMSprop(model.parameters(), lr = 0.001)
model.zero_grad()

var_x, var_y, lengths, xtest, ytest, test_lens = make_r2rt_data(20, 1000, 1)
## manually run model a few iterations


bsiz = 20 # mini-batches of 50 sequences at a time
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
