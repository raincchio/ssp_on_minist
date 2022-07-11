with open("../loss/adam_test_1", 'r') as f:
    data = f.read()
    adam_1 = list(map(eval, data.split("\n")[:-1]))

with open("../loss/adam_test_2", 'r') as f:
    data = f.read()
    adam_2 = list(map(eval, data.split("\n")[:-1]))


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
    for loss in loss_data:
        train_loss.extend(loss)

    # return smooth(train_loss,20)
    return train_loss



import matplotlib.pyplot as plt


adam_1 = process_data(adam_1)

adam_2 = process_data(adam_2)


fig, ax = plt.subplots(1,3)

# first row
figa = ax
figa.plot(adam_1, color='b',label='adm1(lr=0.01)')
figa.plot(adam_2, color='r',label='adm2(lr=0.01)')
xpoint = [0,20,40,60,80,100,120,140,160,180,200]
xvalue = [469*x for x in xpoint]
figa.set_xticks(xvalue)
figa.set_xticklabels(xpoint)
figa.set_title('train loss')
figa.legend()






figa.set_ylabel('loss')

plt.tight_layout()
plt.show()
# plt.savefig("../figure/together")

# print(adam_loss)