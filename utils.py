import math
import torch
from torch import Tensor
from typing import List, Optional




def compute_numer(B1, B2):
    result = []

    for pg in B1[0][0]:
        pg_ = []
        for para in pg:
            pg_.append(torch.zeros_like(para))
        result.append(pg_)

    for idx in range(len(B1)):
        # param index is 0, param is coompose by group
        for idx_pg in range(len(B1[idx][0])): # only one dimension
            for idx_param, param in enumerate(B1[idx][0][idx_pg]):
                result[idx_pg][idx_param] += B1[idx][1][idx_pg][idx_param]*(param-B2[idx][0][idx_pg][idx_param])

    return result

def compute_denom(B1):
    result = []
    for pg in B1[0][0]:
        pg_ = []
        for para in pg:
            pg_.append(torch.zeros_like(para))
        result.append(pg_)

    for param_grad_data in B1:
        for idx_pg, grad_group in enumerate(param_grad_data[1]):  # only one dimension
            for idx_grad, grad in enumerate(grad_group):
                result[idx_pg][idx_grad] += grad * grad

    return result

def check_value(nu,de):
    # result = torch.zeros_like(nu)
    for idx_pg in range(len(nu)):
        for idx in range(len(nu[idx_pg])):
            fmask= (de[idx_pg][idx]==0)
            de[idx_pg][idx][fmask] = 1
            nu[idx_pg][idx][fmask] = 0
            nu[idx_pg][idx] /= de[idx_pg][idx]
    # for nu_g, de_g in zip(nu,de):
    #     for nu_item, de_item in zip(nu_g, de_g):
    #         fmask = (de_item == 0)
    #         de_item[fmask] = 1
    #         nu_item[fmask] = 0
    #         nu_item = nu_item / de_item

    # for i in range(len(de)):
    #     fmask = (de[i]==0)
    #     de[i][fmask] = 1
    #     nu[i][fmask] = 0
    #     nu[i] = nu[i]/de[i]
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
