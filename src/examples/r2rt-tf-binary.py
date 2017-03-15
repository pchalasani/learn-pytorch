from rnn import *

# best settings:
#model = RNN(rnn_type ='GRU', nh=20, nlay=2, dropout = 0.5)
model = RNN(rnn_type ='GRU', nh=50, nlay=32, dropout = 0.7)
#loss_fn = nn.MSELoss(size_average=True)
loss_fn = nn.BCELoss(size_average=True)
optimizer = optim.Adam(model.parameters(), lr = 5e-4)
#optimizer = optim.Adam(model.parameters(), lr = 0.7)
#optimizer = optim.RMSprop(model.parameters(), lr = 0.001)
model.zero_grad()

var_x, var_y, lengths, xtest, ytest, test_lens = make_r2rt_data(500, 10, 1)
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
