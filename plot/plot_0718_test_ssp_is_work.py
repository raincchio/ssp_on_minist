# we do experiments about ssp's effect
# three comparative expeimnts

#

# load data

with open("../loss/0714/gd-sgd-lr_0.01-K_3", 'r') as f:
    data = f.read()
    data = data.replace("nan", '0.5')
    sgd_gd = list(map(eval, data.split("\n")[:-1]))

with open("../loss/0714/sgd-lr_0.01", 'r') as f:
    data = f.read()
    data = data.replace("nan", '0.5')
    sgd = list(map(eval, data.split("\n")[:-1]))

with open("../loss/0714/ssp-sgd-lr_0.01-K_3", 'r') as f:
    data = f.read()
    sgd_ssp = list(map(eval, data.split("\n")[:-1]))


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
    return train_loss, test_loss
    # return smooth(train_loss,469), test_loss



import matplotlib.pyplot as plt

# 0.01 lr
sgd_train_loss, sgd_test_loss = process_data(sgd)
gd_train_loss, gd_test_loss = process_data(sgd_gd)
ssp_train_loss, ssp_test_loss = process_data(sgd_ssp)

fig, ax = plt.subplots()
clrs= ['#4c72b0','#55a868','#c44e52','#8172b2','#ccb974']
# first row
figa = ax
figa.plot(sgd_train_loss, color=clrs[0],label='sgd(lr=0.01)')
# figa.plot(ssp_01_1_train_loss, color=clrs[1],label='sgd+ssp(lr=0.1,k=1)')
figa.plot(gd_train_loss, color=clrs[2],label='sgd+gd(lr=0.01,k=3)')
# figa.plot(ssp_train_loss, color=clrs[3],label='sgd+ssp(lr=0.01,k=3)')

# figa.set_yscale('log')
figa.set_title('train loss(a)')
figa.legend()


figa.set_ylabel('loss')

plt.tight_layout()
plt.show()
# plt.savefig("../figure/0714")

# print(adam_loss)