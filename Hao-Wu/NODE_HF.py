from torchdiffeq import odeint

import math
import os
import torch
import numpy as np
from torch.autograd import Variable
import torch.nn as nn
import copy
import torch.nn.functional as F

def standard_normal_logprob(z):
    logZ = -0.5 * math.log(2 * math.pi)
    return logZ - z.pow(2) / 2


def set_cnf_options(args, model):

    def _set(module):
        if isinstance(module, CNF):
            # Set training settings
            module.solver = args.solver
            module.atol = args.atol
            module.rtol = args.rtol
            if args.step_size is not None:
                module.solver_options['step_size'] = args.step_size

            # If using fixed-grid adams, restrict order to not be too high.
            if args.solver in ['fixed_adams', 'explicit_adams']:
                module.solver_options['max_order'] = 4

            # Set the test settings
            module.test_solver = args.test_solver if args.test_solver else args.solver
            module.test_atol = args.test_atol if args.test_atol else args.atol
            module.test_rtol = args.test_rtol if args.test_rtol else args.rtol

        if isinstance(module, ODEfunc):
            module.rademacher = args.rademacher
            module.residual = args.residual

    model.apply(_set)


# def override_divergence_fn(model, divergence_fn):

#     def _set(module):
#         if isinstance(module, layers.ODEfunc):
#             if divergence_fn == "brute_force":
#                 module.divergence_fn = divergence_bf
#             elif divergence_fn == "approximate":
#                 module.divergence_fn = divergence_approx

#     model.apply(_set)


def count_nfe(model):

    class AccNumEvals(object):

        def __init__(self):
            self.num_evals = 0

        def __call__(self, module):
            if isinstance(module, ODEfunc):
                self.num_evals += module.num_evals()

    accumulator = AccNumEvals()
    model.apply(accumulator)
    return accumulator.num_evals


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_total_time(model):

    class Accumulator(object):

        def __init__(self):
            self.total_time = 0

        def __call__(self, module):
            if isinstance(module, CNF):
                self.total_time = self.total_time + module.sqrt_end_time * module.sqrt_end_time

    accumulator = Accumulator()
    model.apply(accumulator)
    return accumulator.total_time


def build_ODE_model_tabular(args, dims, diffeq):

    hidden_dims = tuple(map(int, args.dims.split("-")))

    def build_cnf():
#         diffeq = layers.ODEnet(
#             hidden_dims=hidden_dims,
#             input_shape=(dims,),
#             strides=None,
#             conv=False,
#             layer_type=args.layer_type,
#             nonlinearity=args.nonlinearity,
#         )
        odefunc = ODEfunc(
            diffeq=diffeq,
            divergence_fn=args.divergence_fn,
            residual=args.residual,
            rademacher=args.rademacher,
        )
        cnf = CNF(
            odefunc=odefunc,
            T=args.time_length,
            train_T=args.train_T,
            solver=args.solver,
        )
        return cnf

    chain = [build_cnf() for _ in range(args.num_blocks)]

    model = SequentialFlow(chain)

    set_cnf_options(args, model)

    return model


class SequentialFlow(nn.Module):
    """A generalized nn.Sequential container for normalizing flows.
    """

    def __init__(self, layersList):
        super(SequentialFlow, self).__init__()
        self.chain = nn.ModuleList(layersList)

    def forward(self, x, logpx=None, reverse=False, inds=None):
        if inds is None:
            if reverse:
                inds = range(len(self.chain) - 1, -1, -1)
            else:
                inds = range(len(self.chain))

        if logpx is None:
            for i in inds:
                x = self.chain[i](x, reverse=reverse)
            return x
        else:
            for i in inds:
                x, logpx = self.chain[i](x, logpx, reverse=reverse)
            return x, logpx

        
        
def get_transforms(model):

    def sample_fn(z, logpz=None):
        if logpz is not None:
            return model(z, logpz, reverse=True)
        else:
            return model(z, reverse=True)

    def density_fn(x, logpx=None):
        if logpx is not None:
            return model(x, logpx, reverse=False)
        else:
            return model(x, reverse=False)

    return sample_fn, density_fn



class CNF(nn.Module):
    def __init__(self, odefunc, T=1.0, train_T=False, solver='dopri5', atol=1e-5, rtol=1e-5):
        super(CNF, self).__init__()
        if train_T:
            self.register_parameter("sqrt_end_time", nn.Parameter(torch.sqrt(torch.tensor(T))))
        else:
            self.register_buffer("sqrt_end_time", torch.sqrt(torch.tensor(T)))

        self.odefunc = odefunc
        self.solver = solver
        self.atol = atol
        self.rtol = rtol
        self.test_solver = solver
        self.test_atol = atol
        self.test_rtol = rtol
        self.solver_options = {}

    def forward(self, z, logpz=None, integration_times=None, reverse=False):

        if logpz is None:
            _logpz = torch.zeros(z.shape[0], 1).to(z)
        else:
            _logpz = logpz

        if integration_times is None:
            integration_times = torch.tensor([0.0, self.sqrt_end_time * self.sqrt_end_time]).to(z)
        if reverse:
            integration_times = _flip(integration_times, 0)

        # Refresh the odefunc statistics.
        self.odefunc.before_odeint()
#         print("z:", z.device)
#         print("log rho:", _logpz.device)
#         print("time: ", integration_times.to(z).device)
        if self.training:
            state_t = odeint(
                self.odefunc,
                (z, _logpz),# + reg_states,
                integration_times.to(z),
                atol=self.atol,
                rtol=self.rtol,
                method=self.solver,
                options=self.solver_options,
            )
        else:
            state_t = odeint(
                self.odefunc,
                (z, _logpz),
                integration_times.to(z),
                atol=self.test_atol,
                rtol=self.test_rtol,
                method=self.test_solver,
            )
        #print("state:", state_t)

        if len(integration_times) == 2:
            state_t = tuple(s[1] for s in state_t)
            
        z_t, logpz_t = state_t[:2]
        return z_t, logpz_t
#         else:
#             return z_t

#     def get_regularization_states(self):
#         reg_states = self.regularization_states
#         self.regularization_states = None
#         return reg_states

    def num_evals(self):
        return self.odefunc._num_evals.item()


def _flip(x, dim):
    indices = [slice(None)] * x.dim()
    indices[dim] = torch.arange(x.size(dim) - 1, -1, -1, dtype=torch.long, device=x.device)
    return x[tuple(indices)]



def divergence_bf(dx, y, **unused_kwargs):
    sum_diag = 0.
    for i in range(y.shape[1]):
        sum_diag += torch.autograd.grad(dx[:, i].sum(), y, create_graph=True)[0].contiguous()[:, i].contiguous()
    return sum_diag.contiguous()


# def divergence_bf(f, y, **unused_kwargs):
#     jac = _get_minibatch_jacobian(f, y)
#     diagonal = jac.view(jac.shape[0], -1)[:, ::jac.shape[1]]
#     return torch.sum(diagonal, 1)


def _get_minibatch_jacobian(y, x):
    """Computes the Jacobian of y wrt x assuming minibatch-mode.

    Args:
      y: (N, ...) with a total of D_y elements in ...
      x: (N, ...) with a total of D_x elements in ...
    Returns:
      The minibatch Jacobian matrix of shape (N, D_y, D_x)
    """
    assert y.shape[0] == x.shape[0]
    y = y.view(y.shape[0], -1)

    # Compute Jacobian row by row.
    jac = []
    for j in range(y.shape[1]):
        dy_j_dx = torch.autograd.grad(y[:, j], x, torch.ones_like(y[:, j]), retain_graph=True,
                                      create_graph=True)[0].view(x.shape[0], -1)
        jac.append(torch.unsqueeze(dy_j_dx, 1))
    jac = torch.cat(jac, 1)
    return jac


def divergence_approx(f, y, e=None):
    e_dzdx = torch.autograd.grad(f, y, e, create_graph=True)[0]
    e_dzdx_e = e_dzdx * e
    approx_tr_dzdx = e_dzdx_e.view(y.shape[0], -1).sum(dim=1)
    return approx_tr_dzdx


def sample_rademacher_like(y):
    return torch.randint(low=0, high=2, size=y.shape).to(y) * 2 - 1


def sample_gaussian_like(y):
    return torch.randn_like(y)


class ODEnet(nn.Module):
    pass



class ODEfunc(nn.Module):

    def __init__(self, diffeq, divergence_fn="approximate", residual=False, rademacher=False):
        super(ODEfunc, self).__init__()
        assert divergence_fn in ("brute_force", "approximate")

        # self.diffeq = diffeq_layers.wrappers.diffeq_wrapper(diffeq)
        self.diffeq = diffeq
        self.residual = residual
        self.rademacher = rademacher

        if divergence_fn == "brute_force":
            self.divergence_fn = divergence_bf
        elif divergence_fn == "approximate":
            self.divergence_fn = divergence_approx

        self.register_buffer("_num_evals", torch.tensor(0.))

    def before_odeint(self, e=None):
        self._e = e
        self._num_evals.fill_(0)

    def num_evals(self):
        return self._num_evals.item()

    def forward(self, t, states):
        assert len(states) >= 2
        y = states[0]

        # increment num evals
        self._num_evals += 1

        # convert to tensor
        t = torch.tensor(t).type_as(y)
        batchsize = y.shape[0]

        # Sample and fix the noise.
        if self._e is None:
            if self.rademacher:
                self._e = sample_rademacher_like(y)
            else:
                self._e = sample_gaussian_like(y)

        with torch.set_grad_enabled(True):
            y.requires_grad_(True)
            t.requires_grad_(True)
            for s_ in states[2:]:
                s_.requires_grad_(True)
            dy = self.diffeq(t, y, *states[2:])
            # Hack for 2D data to use brute force divergence computation.
            divergence = divergence_bf(dy, y).view(batchsize, 1)
#             if not self.training and dy.view(dy.shape[0], -1).shape[1] == 2:
#                 divergence = divergence_bf(dy, y).view(batchsize, 1
#             else:
#                 divergence = self.divergence_fn(dy, y, e=self._e).view(batchsize, 1)
        if self.residual:
            dy = dy - y
            divergence -= torch.ones_like(divergence) * torch.tensor(np.prod(y.shape[1:]), dtype=torch.float32
                                                                     ).to(divergence)
        return tuple([dy, -divergence] + [torch.zeros_like(s_).requires_grad_(True) for s_ in states[2:]])


    
    
def build_model(params):
    layers = []
    layers.append(nn.Linear(params['dim'], params['hlayers'][0]))
    num_of_layers = len(params['hlayers'])

    for h in range(num_of_layers - 1):
        af = params['hactivations'][h]
        if af.lower() != 'none':
            layers.append(build_activation_function(af))
        layers.append(nn.Linear(params['hlayers'][h], params['hlayers'][h+1]))
    af = params['hactivations'][num_of_layers-1]
    if af.lower() != 'none':
        layers.append(build_activation_function(af))
    layers.append(nn.Linear(params['hlayers'][num_of_layers-1], params['dim'], bias=False))

    b = params['init_b']
    if b!=None:
        for l in layers:
            # if the length of network is large, then the value of wights should be near 1.0,
            # otherwise the value of derivative of the network will be small (<< 1.0)
            if isinstance(l, nn.Linear):
                l.weight.data.uniform_(-b, b)
                if l.bias!=None:
                    l.bias.data.uniform_(-b, b)
    return nn.Sequential(*layers)

def build_activation_function(af):
    exp_af = 'lambda _ : nn.' + af
    return eval(exp_af)(None)
  

class MLP(nn.Module):
    def __init__(self, params):
        super(MLP, self).__init__()
        self.map = build_model(params).to(params['device'])

    def forward(self, t, x):
        return self.map(x)#, None
    
    

