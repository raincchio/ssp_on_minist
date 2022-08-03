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


## run ssp in stochastic gradient
in this setting, we use the stochastice gradient to comupte the alpha.
Coment on the SSP.step line to compare the results.
Also can use exp_average flag to control if exp averager over the histroy gradient.
```bash
python run_ssp_in_sgd_update.py 
```
