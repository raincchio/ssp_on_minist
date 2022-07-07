with open("../loss/sgd_loss", 'r') as f:
    data = f.read()
    sgd_loss = list(map(eval, data.split("\n")[:-1]))

with open("../loss/ssp+sgd_in_epoch_loss", 'r') as f:
    data = f.read()
    ssp_sgd_loss = list(map(eval, data.split("\n")[:-1]))


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


import matplotlib.pyplot as plt

sgd_loss = np.array(sgd_loss).flatten()
ssp_sgd_loss = np.array(ssp_sgd_loss).flatten()
fig, ax = plt.subplots()

ax.plot(smooth(sgd_loss, 100), color='b',label='sgd(lr=0.001)')
ax.plot(smooth(ssp_sgd_loss, 100), color='r',label='sgd+ssp(lr=0.001)')
ax.set_title('sgd vs sgd+ssp')

ax.set_xlabel('epoch')

xpoint = [0,20,40,60,80,100,120,140,160,180,200]
xvalue = [469*x for x in xpoint]
ax.set_xticks(xvalue)
ax.set_xticklabels(xpoint)
ax.set_ylabel('loss')

ax.legend()
# plt.show()
plt.savefig("../figure/sspvssgd")

# print(adam_loss)