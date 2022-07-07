with open("../loss/adam_loss", 'r') as f:
    data = f.read()
    adam_loss = list(map(eval, data.split("\n")[:-1]))

with open("../loss/ssp+sgd_in_epoch_loss_lr_1", 'r') as f:
    data = f.read()
    ssp_adam_loss = list(map(eval, data.split("\n")[:-1]))


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

adam_loss = np.array(adam_loss).flatten()
ssp_adam_loss = np.array(ssp_adam_loss).flatten()
fig, ax = plt.subplots()

ax.plot(smooth(adam_loss, 469), color='b',label='adam(lr=0.001)')
ax.plot(smooth(ssp_adam_loss, 469), color='r',label='adam+ssp(lr=0.001)')
ax.set_title('adam vs adam+ssp')
# (step-size plan in every 2 epoch)

ax.set_xlabel('epoch')

xpoint = [0,20,40,60,80,100,120,140,160,180,200]
xvalue = [469*x for x in xpoint]
ax.set_xticks(xvalue)
ax.set_xticklabels(xpoint)
ax.set_ylabel('loss')

ax.legend()
plt.savefig("../figure/sspvsadam___test")

# print(adam_loss)