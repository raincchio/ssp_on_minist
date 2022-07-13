import torch
# from torch.optim import optimizer
from collections import defaultdict
import torch.nn.functional as F
import utils as U
from torchvision import datasets, transforms
from copy import deepcopy


class SSP(object):
    def __init__(self):
        self.state = defaultdict(dict)
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
                    grads_.append(self.state[p]['exp_avg_grad'])
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
                    if copymethod=="ref":
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

    def step_with_true_gradient(self, model, device, dataset, train_kwargs, optimizer, epoch, K=2, sampledata=False, noise=False):
        # one step sgd
        if sampledata:
            grad_groupslist = []
            train_kwargs_ = deepcopy(train_kwargs)
            train_kwargs_["batch_size"] = 10000
            train_loader = torch.utils.data.DataLoader(dataset, **train_kwargs_)
            for batch_idx, (data, target) in enumerate(train_loader):

                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = model(data)
                loss = F.nll_loss(output, target)
                loss.backward()
                grad_groupslist.append(self.getGrad(optimizer))

                # if len(grad_groupslist)==samplesize:
                    # print(len(grad_groupslist),samplesize)
                    # break
            # print(len(grad_groupslist), samplesize)
            # assert  == samplesize, "error, check count variable"
            samplesize = len(grad_groupslist)

            for grad_groups in grad_groupslist[:-1]:
                for gidx, grad_group in enumerate(grad_groups):
                    for idx, grad in enumerate(grad_groupslist[-1][gidx]):
                        grad += grad_group[idx]

            for grad_group in grad_groupslist[-1]:
                for grad in grad_group:
                    grad /= samplesize

            true_grads = grad_groupslist[-1]
            # compute average

        else:
            train_kwargs['batch_size'] = len(dataset)
            train_loader = torch.utils.data.DataLoader(dataset, **train_kwargs)
            data, target = next(iter(train_loader))
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            true_grads = self.getGrad(optimizer)

        params = self.getWeight(optimizer)

        # self.Buffer.append([params, true_grads])
        #
        # if len(self.Buffer2) == K*2:
        #
        #     grads = self.Buffer[-1]
        #
        #     with torch.no_grad():
        #
        #     # compute alpha from buffer
        #         numerator = U.compute_numer(self.Buffer[:K], self.Buffer[K:])
        #         denominator = U.compute_denom(self.Buffer[:K])
        #
        #         alpha = U.check_value(numerator, denominator)
        #
        #         # update parameters
        #         # for pg in zipparams:
        #         params = self.getWeight(optimizer,copymethod='ref')
        #         for idx_pg in range(len(params)):
        #             for i, param in enumerate(params[idx_pg]):
        #                 param.add_(-alpha[idx_pg][i]*grads[idx_pg][i])
        #
        #         self.Buffer.pop(0)


        if epoch <= K:
            self.Buffer1.append([params, true_grads])
        else:
            self.Buffer2.append([params, true_grads])

        if len(self.Buffer2) == K:

            _, grads = self.Buffer2[-1]

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

            self.Buffer1 = self.Buffer2 + []
            self.Buffer2.clear()
            print("do step size planning")


    def step_with_exp_average_gradient(self, batch_idx, optimizer, buffersize=2):
        # one step sgd
        params = self.getWeight(optimizer)
        grads = self.getGrad(optimizer)
        # resotre param

        U.update_exp_average_grad(optimizer)
        average_grads = self.getExpGrad(optimizer)

        with torch.no_grad():
            if batch_idx < buffersize:
                self.Buffer1.append([params, average_grads])
            else:
                self.Buffer2.append([params, average_grads])

            if len(self.Buffer2) == buffersize:
                # compute

                # compute alpha from buffer
                numerator = U.compute_numer(self.Buffer1, self.Buffer2)
                denominator = U.compute_denom(self.Buffer1)

                alpha = U.check_value(numerator, denominator)

                # update parameters
                for i, param in enumerate(params):
                    param.add_(-alpha[i]*grads[i])

                self.Buffer1 = self.Buffer2 + []
                self.Buffer2.clear()