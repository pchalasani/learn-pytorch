from rnn import *
import matplotlib.pyplot as plt

# best settings:
#model = RNN(rnn_type ='GRU', nh=20, nlay=2, dropout = 0.5)
model = RNN(rnn_type ='GRU', nf=2, nh=20, nlay=2, dropout = 0.2)

# BEST SETTING SO FAR!! AChieves loss 0f close to 0.45 !!
#model = RNN(rnn_type ='LSTM', nf=2, nh=30, nlay=1, dropout = 0.2)

#loss_fn = nn.MSELoss(size_average=True)
loss_fn = nn.BCELoss(size_average=True)
optimizer = optim.Adam(model.parameters(), lr = 5e-4)
#optimizer = optim.Adam(model.parameters(), lr = 0.7)
#optimizer = optim.RMSprop(model.parameters(), lr = 0.001)
model.zero_grad()

def gen(x):
    threshold = 0.5
    if x[5] == 1:
        threshold += 0.5
    if x[0] == 1:
        threshold -= 0.25
    if (np.random.rand() > threshold):
        return 0
    else:
        return 1

#var_x, var_y, lengths, xtest, ytest, test_lens = make_r2rt_data(25, 400, 1, xhot = True)
var_x, var_y, lengths, xtest, ytest, test_lens = make_seq_data(gen, 8, 25, 400, 1, xhot = True)
## manually run model a few iterations


bsiz = 50 # mini-batches of 50 sequences at a time
nb = var_x.data.size()[1]/bsiz  # how many mini-batches per epoch

fig = plt.figure()
ax = fig.add_subplot(111)
plt.ion()

fig.show()
fig.canvas.draw()
nepochs = 60
points = np.zeros(nepochs) # train losses
val_points = np.zeros(nepochs) # validation losses

for epoch in range(nepochs): #tqdm(range(500)):
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
    points[epoch] = loss.data[0]
    val_loss = loss_fn(model(xtest, None)[0], ytest).data[0]
    val_points[epoch] = val_loss
    ax.clear()
    ax.plot(points)
    ax.plot(val_points)
    fig.canvas.draw()

    # print 'loss = ', loss.data[0]
    # ny = min(bsiz,15)
