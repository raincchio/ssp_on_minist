import time

import torch
# from torch.optim import optimizer
from collections import defaultdict
import torch.nn.functional as F
import utils as U
from torchvision import datasets, transforms
from copy import deepcopy


class SSP(object):
    def __init__(self):
        self.Buffer1 = []
        self.Buffer2 = []
        self.Buffer = []
        self.average_g_h = []

    def __setstate__(self, state):
        self.state = state

    def getExpGrad(self, optimizer):
        grads = []

        for group in optimizer.param_groups:
            grads_ = []
            for p in group['params']:
                if p.grad is not None:
                    grads_.append(optimizer.state[p]['exp_avg_grad'])
            grads.append(grads_)

        return grads

    def getWeightAndGrad(self, optimizer):
        params = []
        grads = []
        for group in optimizer.param_groups:
            params_ = []
            grads_ = []

            for p in group['params']:
                if p.grad is not None:
                    p_ = p.clone()
                    params_.append(p_)
                    grads_.append(p_.grad)
            params.append(params_)
            grads.append(grads_)

        return params, grads

    def getGrad(self, optimizer):
        grads = []

        for group in optimizer.param_groups:
            grads_ = []
            for p in group['params']:
                if p.grad is not None:
                    grads_.append(p.grad.clone())
            grads.append(grads_)

        return grads

    def getWeight(self, optimizer, copymethod='value'):
        '''

        :param optimizer:
        :param copymethod: value or reference
        :return:
        '''

        params = []
        for group in optimizer.param_groups:
            params_ = []
            for p in group['params']:
                if p.grad is not None:
                    if copymethod == "ref":
                        params_.append(p)
                    else:
                        params_.append(p.clone())
            params.append(params_)

        return params

    @torch.no_grad()
    def step_reserve(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            params_with_grad = []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []
            state_steps = []
            for p in group['params']:
                if p.grad is not None:
                    params_with_grad.append(p)
                    grads.append(p.grad)
                    state = self.state[p]
                    # Lazy state initialization
                    if len(state) == 0:
                        state['step'] = 0
                        # Exponential moving average of gradient values
                        state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        # Exponential moving average of squared gradient values
                        state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                    exp_avgs.append(state['exp_avg'])
                    exp_avg_sqs.append(state['exp_avg_sq'])

                    # update the steps for each param group update
                    state['step'] += 1
                    # record the step after step update
                    state_steps.append(state['step'])
        return loss

    def step_with_true_gradient(self, model, device, dataset, optimizer, epoch, fpath, K=2,
                                noise=False):
        # one step sgd
        train_kwargs ={
            "batch_size": 15000,
            'num_workers': 0,
            'pin_memory': False,
            'shuffle': False
        }
        dataloader = torch.utils.data.DataLoader(dataset, **train_kwargs)

        optimizer.zero_grad()
        loss = 0
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss += F.nll_loss(output, target)
        (loss/len(dataloader)).backward()

        true_grads = self.getGrad(optimizer)
        params = self.getWeight(optimizer)

        if len(self.Buffer1) < K:
            self.Buffer1.append([params, true_grads])
        else:
            self.Buffer2.append([params, true_grads])

        if len(self.Buffer2) == K:

            _, grads = self.Buffer2[-1]
            #
            with torch.no_grad():

                # compute alpha from buffer
                numerator = U.compute_numer(self.Buffer1, self.Buffer2)
                denominator = U.compute_denom(self.Buffer1, noise)

                alpha = U.check_value(numerator, denominator)

                # update parameters
                # note: using params in the optimizer

                params = self.getWeight(optimizer,copymethod='ref')
                for idx_pg in range(len(params)):
                    for i, param in enumerate(params[idx_pg]):
                        param.add_(-alpha[idx_pg][i]*grads[idx_pg][i])

                # torch.save(alpha, fpath+'alpha/alpha-'+str(epoch))

            self.Buffer1 = self.Buffer2 + []
            # self.Buffer1.clear()
            self.Buffer2.clear()
            print("do step size planning")

    def test_gradient(self, model, device, dataset, optimizer, K=2, lr=0.01):


        if len(self.Buffer1) < K:
            self.Buffer1.append([1])
        else:
            self.Buffer2.append([1])

        if len(self.Buffer2) == K:

            train_kwargs = {
                "batch_size": 15000,
                'num_workers': 0,
                'pin_memory': False,
                'shuffle': False
            }
            dataloader = torch.utils.data.DataLoader(dataset, **train_kwargs)
            for data, target in dataloader:
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = model(data)
                loss = F.nll_loss(output, target)
                loss.backward()

            grads = self.getGrad(optimizer)
            #
            with torch.no_grad():

                params = self.getWeight(optimizer, copymethod='ref')
                for idx_pg in range(len(params)):
                    for i, param in enumerate(params[idx_pg]):
                        param.add_(grads[idx_pg][i], alpha=-lr)

            self.Buffer2.clear()
            print("do gd instead of step size planning")
    def step_with_stochastic_gradient(self, model, data, batch_idx, optimizer, buffersize=2, exp_average=False):
        # one step sgd
        params = self.getWeight(optimizer)
        # stochastic grads
        grads = self.getGrad(optimizer)
        # resotre param
        if exp_average:
            U.update_exp_average_grad(optimizer)
            grads = self.getExpGrad(optimizer)

        with torch.no_grad():
            if len(self.Buffer1) < buffersize:
                self.Buffer1.append([params, grads])
            else:
                self.Buffer2.append([params, grads])

            if len(self.Buffer2) == buffersize:
                # compute

                # compute alpha from buffer
                numerator = U.compute_numer(self.Buffer1, self.Buffer2)
                denominator = U.compute_denom(self.Buffer1)

                alpha = U.check_value(numerator, denominator)

                # update parameters
                params = self.getWeight(optimizer, copymethod='ref')
                for idx_pg in range(len(params)):
                    for i, param in enumerate(params[idx_pg]):
                        param.add_(-alpha[idx_pg][i] * grads[idx_pg][i])

                self.Buffer1 = self.Buffer2 + []
                self.Buffer2.clear()
                print('do step size planing')

