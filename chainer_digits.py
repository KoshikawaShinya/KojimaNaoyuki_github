import chainer
import chainer.links as L
import chainer.functions as F
import numpy as np
import matplotlib.pyplot as plt
from chainer import optimizers, Sequential
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split


x, t = load_digits(return_X_y=True)

x = x.astype('float32')
t = t.astype('int32')

x_train, x_test, t_train, t_test = train_test_split(x, t, test_size=0.3)

n_input = 64
n_hidden = 100
n_out = 10
epoch = 100
batch_size = 20

iter_num = 2000
iteration = 0
epoch_num = 0
train_size = x_train.shape[0]

net = Sequential(
    L.Linear(n_input, n_hidden), F.relu,
    L.Linear(n_hidden, n_hidden), F.relu,
    L.Linear(n_hidden, n_out)
)

optimizer = optimizers.MomentumSGD()
optimizer.setup(net)

log_train = {'loss': [], 'accuracy': []}
log_test = {'loss': [], 'accuracy': []}

for i in range(iter_num):

    loss_list = []
    accuracy_list = []

    batch_mask = np.random.choice(train_size, batch_size)
    x_train_batch = x_train[batch_mask]
    t_train_batch = t_train[batch_mask]

    y_train_batch = net(x_train_batch)

    loss_train_batch = F.softmax_cross_entropy(y_train_batch, t_train_batch)
    accuracy_train_batch = F.accuracy(y_train_batch, t_train_batch)

    loss_list.append(loss_train_batch.array)
    accuracy_list.append(accuracy_train_batch.array)

    net.cleargrads()
    loss_train_batch.backward()

    optimizer.update()

    loss_train = np.mean(loss_list)
    accuracy_train = np.mean(accuracy_list)


    if i % epoch == 0:
        epoch_num += 1
        y_test = net(x_test)

        loss_test = F.softmax_cross_entropy(y_test, t_test)
        accuracy_test = F.accuracy(y_test, t_test)

        print('epoch: {}, iteration: {}, loss(train): {:.4f}, loss(test): {:.4f}'.format(epoch_num, iteration, loss_train, loss_test.array))

        log_train['loss'].append(loss_train)
        log_train['accuracy'].append(accuracy_train)
        log_test['loss'].append(loss_test.array)
        log_test['accuracy'].append(accuracy_test.array)

    iteration += 1

plt.plot(log_train['loss'], label='train_loss')
plt.plot(log_test['loss'], label='test_loss')
plt.legend()
plt.show()

plt.plot(log_train['accuracy'], label='train_accuracy')
plt.plot(log_test['accuracy'], label='test_accuracy')
plt.legend()
plt.show()

