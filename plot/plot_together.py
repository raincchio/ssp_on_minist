with open("../loss/sgd_loss", 'r') as f:
    data = f.read()
    sgd_loss = list(map(eval, data.split("\n")[:-1]))

with open("../loss/ssp+sgd_in_epoch_loss", 'r') as f:
    data = f.read()
    ssp_sgd_loss = list(map(eval, data.split("\n")[:-1]))

with open("../loss/ssp+sgd_in_epoch_loss_lr_2", 'r') as f:
    data = f.read()
    ssp_sgd_new = list(map(eval, data.split("\n")[:-1]))

# with open("../loss/ssp+adm_in_epoch_loss", 'r') as f:
#     data = f.read()
#     ssp_adam_loss = list(map(eval, data.split("\n")[:-1]))


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
sgd_loss = smooth(sgd_loss, 10)
ssp_sgd_loss = np.array(ssp_sgd_loss).flatten()
ssp_sgd_loss = smooth(ssp_sgd_loss, 10)
ssp_sgd_new = np.array(ssp_sgd_new).flatten()
ssp_sgd_new = smooth(ssp_sgd_new, 10)
# ssp_adam_loss = np.array(ssp_adam_loss).flatten()

fig, ax = plt.subplots(2,2, sharex=True, figsize=(10,10))

figa = ax[0][0]
figa.plot(sgd_loss, color='b',label='sgd(lr=0.001)')
figa.plot(ssp_sgd_loss, color='r',label='sgd+ssp(lr=0.001)')
# ax.plot(smooth(adam_loss, 469),label='adam(lr=0.001)')
figa.plot(ssp_sgd_new, color='orange', label='sgd+ssp_new(lr=0.001)')
figa.set_title('together')

figa = ax[0][1]
figa.plot(sgd_loss, color='b',label='sgd(lr=0.001)')
figa.set_title('sgd_loss')

figa = ax[1][0]
figa.plot(ssp_sgd_loss, color='r',label='sgd+ssp(lr=0.001)')
figa.set_title('ssp_sgd_loss')
figa.set_xlabel('epoch')

figa = ax[1][1]

figa.plot(ssp_sgd_new,color='orange',label='sgd+ssp_new(lr=0.001)')
figa.set_title('ssp_sgd_new')
figa.set_xlabel('epoch')



xpoint = [0,20,40,60,80,100,120,140,160,180,200]
xvalue = [469*x for x in xpoint]
figa.set_xticks(xvalue)
figa.set_xticklabels(xpoint)
figa.set_ylabel('loss')

ax[0][0].legend()
plt.tight_layout()
plt.show()
# plt.savefig("../figure/together")

# print(adam_loss)