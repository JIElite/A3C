import math
from collections import deque, namedtuple

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable


def wrap_as_variable(np_array, dtype=np.float32):
    # type conversion
    if np_array.dtype != dtype:
        np_array = np_array.astype(dtype)

    # wrap data as torch Variable
    # TODO support CUDA 
    return Variable(torch.from_numpy(np_array)) 


def init_weights(network):
    for module in network.modules():
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(0.0, 0.01)
            module.bias.data.fill_(0)

        if isinstance(module, nn.Conv2d):
            # TODO need to implement
            pass


def ensure_shared_grad(global_net, local_net):
    for l_parameter, g_parameter in zip(local_net.parameters(), global_net.parameters()):
        g_parameter._grad = l_parameter.grad


# class SharedAdam(torch.optim.Adam):
#     """
#     This code is borrowed from Morvan's pytoch-A3C:
#     https://github.com/MorvanZhou/pytorch-A3C
#     """
#     def __init__(self, params, lr=1e-3, betas=(0.9, 0.9), eps=1e-8,
#                  weight_decay=0):
#         super(SharedAdam, self).__init__(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
#         # State initialization
#         for group in self.param_groups:
#             for p in group['params']:
#                 state = self.state[p]
#                 state['step'] = 0
#                 state['exp_avg'] = torch.zeros_like(p.data)
#                 state['exp_avg_sq'] = torch.zeros_like(p.data)

#                 # share in memory
#                 state['exp_avg'].share_memory_()
#                 state['exp_avg_sq'].share_memory_()


Transition = namedtuple('Transition', ('state', 'action', 'log_action_prob', 'reward'))
class Buffer:
    def __init__(self, size=8):
        self.maxlen = size
        self.memory = deque(maxlen=size)

    def append(self, experience):
        self.memory.append(Transition(*experience))

    def is_full(self):
        return len(self.memory) == self.maxlen 

    def reset(self):
        self.memory.clear()

    def get_reversed_experience(self):
        return reversed(self.memory)


class SharedAdam(torch.optim.Adam):
    """Implements Adam algorithm with shared states.
    """

    def __init__(self,
                 params,
                 lr=1e-3,
                 betas=(0.9, 0.999),
                 eps=1e-8,
                 weight_decay=0):
        super(SharedAdam, self).__init__(params, lr, betas, eps, weight_decay)

        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = torch.zeros(1)
                state['exp_avg'] = p.data.new().resize_as_(p.data).zero_()
                state['exp_avg_sq'] = p.data.new().resize_as_(p.data).zero_()

    def share_memory(self):
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'].share_memory_()
                state['exp_avg'].share_memory_()
                state['exp_avg_sq'].share_memory_()

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad = grad.add(group['weight_decay'], p.data)

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)

                denom = exp_avg_sq.sqrt().add_(group['eps'])

                bias_correction1 = 1 - beta1**state['step'][0]
                bias_correction2 = 1 - beta2**state['step'][0]
                step_size = group['lr'] * math.sqrt(
                    bias_correction2) / bias_correction1

                p.data.addcdiv_(-step_size, exp_avg, denom)

        return loss