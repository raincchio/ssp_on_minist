with open("../loss/0713/sgd-lr_0.01", 'r') as f:
    data = f.read()
    sgd_001 = list(map(eval, data.split("\n")[:-30]))

with open("../loss/0713/sgd-lr_0.1", 'r') as f:
    data = f.read()
    sgd_01 = list(map(eval, data.split("\n")[:-30]))

with open("../loss/0713/ssp-sgd-lr_0.1-K_3", 'r') as f:
    data = f.read()
    ssp_sgd_01_3 = list(map(eval, data.split("\n")[:-30]))

with open("../loss/0713/ssp-sgd-lr_0.1-K_5", 'r') as f:
    data = f.read()
    ssp_sgd_01_5 = list(map(eval, data.split("\n")[:-30]))

with open("../loss/0713/ssp-sgd-lr_0.01-noise", 'r') as f:
    data = f.read()
    ssp_sgd_001_3_noise = list(map(eval, data.split("\n")[:-30]))

with open("../loss/0713/ssp-sgd-lr_0.01", 'r') as f:
    data = f.read()
    ssp_sgd_001_3 = list(map(eval, data.split("\n")[:-30]))

import numpy as np

def smooth(y, radius, mode='two_sided', valid_only=False):
    '''
    Smooth signal y, where radius is determines the size of the window

    mode='twosided':
        average over the window [max(index - radius, 0), min(index + radius, len(y)-1)]
    mode='causal':
        average over the window [max(index - radius, 0), index]

    valid_only: put nan in entries where the full-sized window is not available

    '''
    assert mode in ('two_sided', 'causal')
    if len(y) < 2*radius+1:
        return np.ones_like(y) * y.mean()
    elif mode == 'two_sided':
        convkernel = np.ones(2 * radius+1)
        out = np.convolve(y, convkernel,mode='same') / np.convolve(np.ones_like(y), convkernel, mode='same')
        if valid_only:
            out[:radius] = out[-radius:] = np.nan
    elif mode == 'causal':
        convkernel = np.ones(radius)
        out = np.convolve(y, convkernel,mode='full') / np.convolve(np.ones_like(y), convkernel, mode='full')
        out = out[:-radius+1]
        if valid_only:
            out[:radius] = np.nan
    return out

def process_data(loss_data):
    train_loss = []
    test_loss = []
    for loss in loss_data:
        train_loss.extend(loss[0])
        test_loss.append(loss[1])
    return smooth(train_loss,20), test_loss



import matplotlib.pyplot as plt


sgd_001_train_loss, sgd_001_test_loss = process_data(sgd_001)

sgd_01_train_loss, sgd_01_test_loss = process_data(sgd_01)

ssp_sgd_01_3_train_loss, ssp_sgd_01_3_test_loss = process_data(ssp_sgd_01_3)

ssp_sgd_01_5_train_loss, ssp_sgd_01_5_test_loss = process_data(ssp_sgd_01_5)

ssp_sgd_001_3_noise_train_loss, ssp_sgd_001_3_noise_test_loss = process_data(ssp_sgd_001_3_noise)
ssp_sgd_001_3_train_loss, ssp_sgd_001_3_test_loss = process_data(ssp_sgd_001_3)


fig, ax = plt.subplots(2,2, figsize=(10,10))

# first row
figa = ax[0][0]
figa.plot(sgd_01_train_loss, color='b',label='sgd(lr=0.1)')
figa.plot(ssp_sgd_01_3_train_loss, color='r',label='sgd+ssp(lr=0.1,k=3)')
# figa.plot(ssp_sgd_01_5_train_loss, color='orange',label='sgd+ssp(lr=0.1,k=5)')
xpoint = [0,20,40,60,80,100,120,140,160,180,200]
xvalue = [469*x for x in xpoint]
figa.set_xticks(xvalue)
figa.set_xticklabels(xpoint)
figa.set_yscale('log')
figa.set_title('train loss(a)')
figa.legend()

figa = ax[0][1]
figa.plot(sgd_01_test_loss, color='b',label='sgd(lr=0.01)')
figa.plot(ssp_sgd_01_3_test_loss, color='r',label='sgd+ssp(lr=0.1, k=3)')
figa.plot(ssp_sgd_01_5_test_loss, color='orange',label='sgd+ssp(lr=0.1,k=5)')
xpoint = [0,20,40,60,80,100,120,140,160,180,200]
xvalue = [x for x in xpoint]
figa.set_xticks(xvalue)
figa.set_xticklabels(xpoint)
figa.set_title('test loss(b)')
figa.legend()

figa = ax[1][0]
figa.plot(sgd_001_train_loss, color='b',label='sgd(lr=0.01)')
figa.plot(ssp_sgd_001_3_train_loss, color='r',label='sgd+ssp(lr=0.01)')
figa.plot(ssp_sgd_001_3_noise_train_loss, color='orange',label='sgd+ssp+noise(lr=0.01)')
xpoint = [0,20,40,60,80,100,120,140,160,180,200]
xvalue = [x for x in xpoint]
figa.set_xticks(xvalue)
figa.set_xticklabels(xpoint)
figa.set_title('train loss(c)')
figa.set_yscale('log')
figa.set_xlabel('epoch')
figa.legend()

figa = ax[1][1]
figa.plot(sgd_001_test_loss, color='b',label='sgd(lr=0.01)')
figa.plot(ssp_sgd_001_3_test_loss, color='r',label='sgd+ssp(lr=0.01)')
figa.plot(ssp_sgd_001_3_noise_test_loss, color='orange',label='sgd+ssp+noise(lr=0.01)')
xpoint = [0,20,40,60,80,100,120,140,160,180,200]
xvalue = [x for x in xpoint]
figa.set_xticks(xvalue)
figa.set_xticklabels(xpoint)
figa.set_title('test loss(d)')
figa.set_xlabel('epoch')
figa.legend()


figa.set_ylabel('loss')

plt.tight_layout()
plt.show()
# plt.savefig("../figure/0717")

# print(adam_loss)