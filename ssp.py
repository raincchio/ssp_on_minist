import torch
# from torch.optim import optimizer
from collections import defaultdict
import torch.nn.functional as F
import utils as U

class SSP(object):
    def __init__(self):
        self.state = defaultdict(dict)
        self.Buffer1 = []
        self.Buffer2 = []
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
                    params_.append(p.clone())
                    grads_.append(p.grad.clone())
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
    def getWeight(self, current):
        params = []
        for group in current.param_groups:
            params_ = []
            for p in group['params']:
                if p.grad is not None:
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


    def step_with_true_gradient(self, model, device, dataset, train_kwargs, optimizer, epoch, buffersize=2, sampledata=False, samplesize=2):
        # one step sgd
        params = self.getWeight(optimizer)
        grads = self.getGrad(optimizer)

        if sampledata:
            grad_groupslist = []
            train_loader = torch.utils.data.DataLoader(dataset, **train_kwargs)
            for batch_idx, (data, target) in enumerate(train_loader):

                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = model(data)
                loss = F.nll_loss(output, target)
                loss.backward()
                grad_groupslist.append(self.getGrad(optimizer))

                if len(grad_groupslist)==samplesize:
                    break

            assert len(grad_groupslist) == samplesize, "error, check count variable"

            for grad_groups in grad_groupslist[:-1]:
                for gidx, grad_group in enumerate(grad_groups):
                    for idx, grad in enumerate(grad_groupslist[-1][gidx]):
                        grad += grad_group[idx]

            for grad_group in grad_groupslist[-1]:
                for grad in grad_group:
                    grad /= buffersize


            truth_grads = grad_groupslist[-1]
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
            truth_grads = self.getGrad(optimizer)


        with torch.no_grad():
            if epoch <= buffersize:
                self.Buffer1.append([params, truth_grads])
            else:
                self.Buffer2.append([params, truth_grads])

            if len(self.Buffer2) == buffersize:

                # compute alpha from buffer
                numerator = U.compute_numer(self.Buffer1, self.Buffer2)
                denominator = U.compute_denom(self.Buffer1)

                alpha = U.check_value(numerator, denominator)

                # update parameters
                # for pg in zipparams:
                for idx_pg in range(len(params)):
                    for i, param in enumerate(params[idx_pg]):
                        param.add_(-alpha[idx_pg][i]*grads[idx_pg][i])

                self.Buffer1 = self.Buffer2 + []
                self.Buffer2.clear()

    def step_with_average_gradient(self, batch_idx, optimizer, buffersize=2):
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