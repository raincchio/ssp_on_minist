with open("../loss/0707/sgd_0.01", 'r') as f:
    data = f.read()
    sgd_001 = list(map(eval, data.split("\n")[:-1]))

with open("../loss/0707/sgd_0.001", 'r') as f:
    data = f.read()
    sgd_0001 = list(map(eval, data.split("\n")[:-1]))

with open("../loss/0707/sgd_0.0001", 'r') as f:
    data = f.read()
    sgd_00001 = list(map(eval, data.split("\n")[:-1]))

with open("../loss/0707/ssp_0.01", 'r') as f:
    data = f.read()
    ssp_001 = list(map(eval, data.split("\n")[:-1]))

with open("../loss/0707/ssp_0.001", 'r') as f:
    data = f.read()
    ssp_0001 = list(map(eval, data.split("\n")[:-1]))

with open("../loss/0707/ssp_0.0001", 'r') as f:
    data = f.read()
    ssp_00001 = list(map(eval, data.split("\n")[:-1]))


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

sgd_0001_train_loss, sgd_0001_test_loss = process_data(sgd_0001)

sgd_00001_train_loss, sgd_00001_test_loss = process_data(sgd_00001)

ssp_001_train_loss, ssp_001_test_loss = process_data(ssp_001)

ssp_0001_train_loss, ssp_0001_test_loss = process_data(ssp_0001)

ssp_00001_train_loss, ssp_00001_test_loss = process_data(ssp_00001)

fig, ax = plt.subplots(3,3, figsize=(10,10))

# first row
figa = ax[0][0]
figa.plot(sgd_001_train_loss, color='b',label='sgd(lr=0.01)')
figa.plot(ssp_001_train_loss, color='r',label='sgd+ssp(lr=0.01)')
xpoint = [0,20,40,60,80,100,120,140,160,180,200]
xvalue = [469*x for x in xpoint]
figa.set_xticks(xvalue)
figa.set_xticklabels(xpoint)
figa.set_title('train loss')
figa.legend()

figa = ax[0][1]
figa.plot(sgd_001_train_loss, color='b',label='sgd(lr=0.01)')
figa.plot(ssp_001_train_loss, color='r',label='sgd+ssp(lr=0.01)')
xpoint = [0,20,40,60,80,100,120,140,160,180,200]
xvalue = [469*x for x in xpoint]
figa.set_xticks(xvalue)
figa.set_xticklabels(xpoint)
figa.set_yscale('log')
figa.set_title('train loss(log)')
figa.legend()

figa = ax[0][2]
figa.plot(sgd_001_test_loss, color='b',label='sgd(lr=0.01)')
figa.plot(ssp_001_test_loss, color='r',label='sgd+ssp(lr=0.01)')
xpoint = [0,20,40,60,80,100,120,140,160,180,200]
xvalue = [x for x in xpoint]
figa.set_xticks(xvalue)
figa.set_xticklabels(xpoint)
figa.set_title('test loss')
figa.set_xlabel('epoch')
figa.legend()

### second
figa = ax[1][0]
figa.plot(sgd_0001_train_loss, color='b',label='sgd(lr=0.001)')
figa.plot(ssp_0001_train_loss, color='r',label='sgd+ssp(lr=0.001)')
xpoint = [0,20,40,60,80,100,120,140,160,180,200]
xvalue = [469*x for x in xpoint]
figa.set_xticks(xvalue)
figa.set_xticklabels(xpoint)
figa.set_title('train loss')
figa.legend()

figa = ax[1][1]
figa.plot(sgd_0001_train_loss, color='b',label='sgd(lr=0.001)')
figa.plot(ssp_0001_train_loss, color='r',label='sgd+ssp(lr=0.001)')
xpoint = [0,20,40,60,80,100,120,140,160,180,200]
xvalue = [469*x for x in xpoint]
figa.set_xticks(xvalue)
figa.set_xticklabels(xpoint)
figa.set_yscale('log')
figa.set_title('train loss(log)')
figa.legend()

figa = ax[1][2]
figa.plot(sgd_0001_test_loss, color='b',label='sgd(lr=0.001)')
figa.plot(ssp_0001_test_loss, color='r',label='sgd+ssp(lr=0.001)')
xpoint = [0,20,40,60,80,100,120,140,160,180,200]
xvalue = [x for x in xpoint]
figa.set_xticks(xvalue)
figa.set_xticklabels(xpoint)
figa.set_title('test loss')
figa.set_xlabel('epoch')
figa.legend()
### third row

figa = ax[2][0]
figa.plot(sgd_00001_train_loss, color='b',label='sgd(lr=0.0001)')
figa.plot(ssp_0001_train_loss, color='r',label='sgd+ssp(lr=0.001)')
xpoint = [0,20,40,60,80,100,120,140,160,180,200]
xvalue = [469*x for x in xpoint]
figa.set_xticks(xvalue)
figa.set_xticklabels(xpoint)
figa.set_title('train loss')
figa.legend()

figa = ax[2][1]
figa.plot(sgd_00001_train_loss, color='b',label='sgd(lr=0.0001)')
figa.plot(ssp_00001_train_loss, color='r',label='sgd+ssp(lr=0.0001)')
xpoint = [0,20,40,60,80,100,120,140,160,180,200]
xvalue = [469*x for x in xpoint]
figa.set_xticks(xvalue)
figa.set_xticklabels(xpoint)
figa.set_yscale('log')
figa.set_title('train loss(log)')
figa.legend()

figa = ax[2][2]
figa.plot(sgd_00001_test_loss, color='b',label='sgd(lr=0.01)')
figa.plot(ssp_00001_test_loss, color='r',label='sgd+ssp(lr=0.01)')
xpoint = [0,20,40,60,80,100,120,140,160,180,200]
xvalue = [x for x in xpoint]
figa.set_xticks(xvalue)
figa.set_xticklabels(xpoint)
figa.set_title('test loss')
figa.set_xlabel('epoch')
figa.legend()


figa.set_ylabel('loss')

plt.tight_layout()
plt.show()
# plt.savefig("../figure/together")

# print(adam_loss)