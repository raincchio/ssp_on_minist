with open("../loss/0714/ssp-sgd-lr_0.1-K_1", 'r') as f:
    data = f.read()
    data = data.replace("nan", '0.5')
    ssp_01_1 = list(map(eval, data.split("\n")[:-60]))

with open("../loss/0714/ssp-sgd-lr_0.1-K_3", 'r') as f:
    data = f.read()
    data = data.replace("nan", '0.5')
    ssp_01_3 = list(map(eval, data.split("\n")[:-60]))

with open("../loss/0714/ssp-sgd-lr_0.1-K_5", 'r') as f:
    data = f.read()
    ssp_01_5 = list(map(eval, data.split("\n")[:-60]))

with open("../loss/0714/ssp-sgd-lr_0.01-K_1", 'r') as f:
    data = f.read()
    data = data.replace("nan", '0.5')
    ssp_001_1 = list(map(eval, data.split("\n")[:-60]))

with open("../loss/0714/ssp-sgd-lr_0.01-K_3", 'r') as f:
    data = f.read()
    ssp_001_3 = list(map(eval, data.split("\n")[:-60]))

with open("../loss/0714/ssp-sgd-lr_0.01-K_5", 'r') as f:
    data = f.read()
    ssp_001_5 = list(map(eval, data.split("\n")[:-60]))


with open("../loss/0714/sgd-lr_0.1", 'r') as f:
    data = f.read()
    sgd_01 = list(map(eval, data.split("\n")[:-60]))

with open("../loss/0714/sgd-lr_0.01", 'r') as f:
    data = f.read()
    sgd_001 = list(map(eval, data.split("\n")[:-60]))


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
    return smooth(train_loss,469), test_loss



import matplotlib.pyplot as plt

# 0.1 lr
ssp_01_1_train_loss, ssp_01_1_test_loss = process_data(ssp_01_1)
ssp_01_3_train_loss, ssp_01_3_test_loss = process_data(ssp_01_3)
ssp_01_5_train_loss, ssp_01_5_test_loss = process_data(ssp_01_5)
sgd_01_train_loss, sgd_01_test_loss = process_data(sgd_01)

# 0.01 lr
ssp_001_1_train_loss, ssp_001_1_test_loss = process_data(ssp_001_1)
ssp_001_3_train_loss, ssp_001_3_test_loss = process_data(ssp_001_3)
ssp_001_5_train_loss, ssp_001_5_test_loss = process_data(ssp_001_5)
sgd_001_train_loss, sgd_001_test_loss = process_data(sgd_001)

fig, ax = plt.subplots(2,2, figsize=(10,10))
clrs= ['#4c72b0','#55a868','#c44e52','#8172b2','#ccb974']
# first row
figa = ax[0][0]
figa.plot(sgd_01_train_loss, color=clrs[0],label='sgd(lr=0.1)')
# figa.plot(ssp_01_1_train_loss, color=clrs[1],label='sgd+ssp(lr=0.1,k=1)')
figa.plot(ssp_01_3_train_loss, color=clrs[2],label='sgd+ssp(lr=0.1,k=3)')
figa.plot(ssp_01_5_train_loss, color=clrs[3],label='sgd+ssp(lr=0.1,k=5)')
xpoint = [0,20,40,60,80,100,120,140,160,180,200]
xvalue = [469*x for x in xpoint]
figa.set_xticks(xvalue)
figa.set_xticklabels(xpoint)
figa.set_yscale('log')
figa.set_title('train loss(a)')
figa.legend()

figa = ax[0][1]
figa.plot(sgd_01_test_loss, color=clrs[0],label='sgd(lr=0.1)')
# figa.plot(ssp_01_3_test_loss, color=clrs[1],label='sgd+ssp(lr=0.1, k=1)')
figa.plot(ssp_01_3_test_loss, color=clrs[2],label='sgd+ssp(lr=0.1, k=3)')
figa.plot(ssp_01_5_test_loss, color=clrs[3],label='sgd+ssp(lr=0.1,k=5)')
xpoint = [0,20,40,60,80,100,120,140,160,180,200]
xvalue = [x for x in xpoint]
figa.set_xticks(xvalue)
figa.set_xticklabels(xpoint)
# figa.set_yscale('log')
figa.set_title('test loss(b)')
figa.legend()

figa = ax[1][0]
figa.plot(sgd_001_train_loss, color=clrs[0],label='sgd(lr=0.01)')
# figa.plot(ssp_001_1_train_loss, color=clrs[1],label='sgd+ssp(lr=0.01,k=1)')
figa.plot(ssp_001_3_train_loss, color=clrs[2],label='sgd+ssp(lr=0.01,k=3)')
figa.plot(ssp_001_5_train_loss, color=clrs[3],label='sgd+ssp(lr=0.01,k=5)')
xpoint = [0,20,40,60,80,100,120,140,160,180,200]
xvalue = [x*469 for x in xpoint]
figa.set_xticks(xvalue)
figa.set_xticklabels(xpoint)
figa.set_title('train loss(c)')
figa.set_yscale('log')
figa.set_xlabel('epoch')
figa.legend()

figa = ax[1][1]
figa.plot(sgd_001_test_loss, color=clrs[0],label='sgd(lr=0.01)')
# figa.plot(ssp_001_1_test_loss, color=clrs[1],label='sgd+ssp(lr=0.01,k=1)')
figa.plot(ssp_001_3_test_loss, color=clrs[2],label='sgd+ssp(lr=0.01,k=3)')
figa.plot(ssp_001_5_test_loss, color=clrs[3],label='sgd+ssp(lr=0.01,k=5)')
xpoint = [0,20,40,60,80,100,120,140,160,180,200]
xvalue = [x for x in xpoint]
figa.set_xticks(xvalue)
figa.set_yscale('log')
figa.set_xticklabels(xpoint)
figa.set_title('test loss(d)')
figa.set_xlabel('epoch')
figa.legend()


figa.set_ylabel('loss')

plt.tight_layout()
plt.show()
# plt.savefig("../figure/0714")

# print(adam_loss)