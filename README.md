# ssp_on_mnist
test ssp on minist


## run ssp between epoch
in this setting, we evaluate the gradint on the whole dataset
```bash
python run_ssp_between_epoch.py --ssp --lr=0.01 --K=2
```
To compare if ssp wroks, we can make a step gradient descent in the ssp turn
```bash
pyhton run_gd_between_epoch.py --gd --lr=0.01 --K=2
```

Replace the function "train" with "train_GD" in the run_ssp_between_epoch.py and run_gd_between_epoch.py, we can check the result of using global(true) gradient with ssp.


## run ssp in stochastic gradient
in this setting, we use the stochastice gradient to comupte the alpha.
Coment on the SSP.step line to compare the results.
Also can use exp_average flag to control if exp averager over the histroy gradient.
```bash
python run_ssp_in_sgd_update.py 
```

## Note
something will cause the gradient to be uncertity.
such as the network architecture, Dropout layer introduces randomness, the shuffle flag in the dataset setting, and torch's defalut floating point dtyp

## About compute the global gradient.
test/test_true_gradint, there exits three methods to compute the gradient.
a. compute batch data loss, then backward, average gradients over the batch.
b. sum the batch data loss, then bakcward, take the gradient then divide the batch nums
c. backward bathc dataloss,  take the gradient then divide the batch nums
