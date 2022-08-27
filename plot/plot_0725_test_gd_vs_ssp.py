with open("../loss/0725/gd-sgd-lr_0.1-K_3", 'r') as f:
    data = f.read()
    sgd_gd = list(map(eval, data.split("\n")[:20]))

with open("../loss/0725/ssp-sgd-lr_0.1-K_3", 'r') as f:
    data = f.read()
    sgd_ssp = list(map(eval, data.split("\n")[:20]))


def process_data(loss_data):
    train_loss = []
    test_loss = []
    for loss in loss_data:
        train_loss.append(sum(loss[0])/len(loss[0]))
        test_loss.append(loss[1])
    return train_loss, test_loss


import matplotlib.pyplot as plt

# 0.01 lr
gd_train_loss, gd_test_loss = process_data(sgd_gd)
ssp_train_loss, ssp_test_loss = process_data(sgd_ssp)

fig, ax = plt.subplots(2,figsize=(8,8))
clrs= ['#4c72b0','#55a868','#c44e52','#8172b2','#ccb974']

labels=['sgd+gd(lr=0.1)', 'sgd+ssp(lr=0.1)']

# first row
figa = ax[0]
figa.plot(gd_train_loss, color=clrs[0],label=labels[0])
figa.plot(ssp_train_loss, color=clrs[2],label=labels[1])
figa.set_title('train loss(a)')
figa.legend()
figa.set_ylabel('loss')

xpoint = list(range(5,20,3))
figa.set_xticks(xpoint)
figa.set_xticklabels(xpoint)



figa.grid()


figa = ax[1]
figa.plot(gd_test_loss, color=clrs[0],label=labels[0])
figa.plot(ssp_test_loss, color=clrs[2],label=labels[1])
figa.set_title('test loss(a)')
figa.legend()
xpoint = list(range(5,20,3))
figa.set_xticks(xpoint)
figa.set_xticklabels(xpoint)
figa.grid()


plt.tight_layout()
plt.show()