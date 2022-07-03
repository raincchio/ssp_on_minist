import math
import torch
from torch import Tensor
from typing import List, Optional




def compute_numer(B1, B2):
    tmp_ = {}
    for i, pa in enumerate(B1):
        pb = B2[i]
        for j, param in enumerate(pa[0]):
            if j in tmp_.keys():
                tmp_[j] += pa[1][j]*(param-pb[0][j])
            else:
                tmp_[j] = pa[1][j]*(param-pb[0][j])
    return tmp_

def compute_denom(B1):
    tmp_ = {}
    for i, pa in enumerate(B1):
        for j, grad in enumerate(pa[1]):
            if j in tmp_.keys():
                tmp_[j] += grad * grad
            else:
                tmp_[j] = grad * grad
    return tmp_

def check_value(nu,de):
    for i in range(len(de)):
        fmask = (de[i]==0)
        de[i][fmask] = 1
        nu[i][fmask] = 0
        nu[i] = nu[i]/de[i]
        # print(max(nu[i]), min(nu[i]))
    return nu


def update_exp_average_grad(current):
    for group in current.param_groups:

        grads = []
        exp_avg_grads = []
        state_steps = []

        for p in group['params']:
            if p.grad is not None:
                grads.append(p.grad)

                state = current.state[p]
                # Lazy state initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg_grad'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                exp_avg_grads.append(state['exp_avg_grad'])

                # update the steps for each param group update
                state['step'] += 1
                # record the step after step update
                state_steps.append(state['step'])
        beta1 = 0.9
        for i, param in enumerate(grads):

            grad = grads[i]
            exp_avg_grad = exp_avg_grads[i]
            # step = state_steps[i]
            # bias_correction1 = 1 - beta1 ** step
            # Decay the first and second moment running average coefficient
            exp_avg_grad.mul_(beta1).add_(grad, alpha=1 - beta1)
