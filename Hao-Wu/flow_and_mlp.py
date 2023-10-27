import torch
from torch import optim, nn
from torch.autograd import Variable
from torch.nn import functional as F

from others import copy_net, tensor_to_numpy

def safe_log(z):
    return torch.log(z + 1e-7)  
    
class NormalizingFlow(nn.Module):

    def __init__(self, dim, flow_length):
        super().__init__()

        self.transforms = nn.Sequential(*(
            flow(dim) for _ in range(flow_length)
        ))

        self.log_jacobians = nn.Sequential(*(
            flow_log_det_Jacobian(t) for t in self.transforms
        ))

    def forward(self, z):

        log_jacobians = []
        z_rec = []
        sum_log = torch.zeros(size=z.shape).to(z.device)
        for transform, log_jacobian in zip(self.transforms, self.log_jacobians):
            log_jacobians.append(log_jacobian(z))
            z = transform(z)
            z_rec.append(z)
            #print('value of z:', z)
            #print(z_rec)
            sum_log += log_jacobian(z)

        zk = z

        return zk, sum_log#z_rec#log_jacobians

class flow(nn.Module):

    def __init__(self, dim):
        super().__init__()

        self.weight = nn.Parameter(torch.Tensor(1, dim))
        self.bias = nn.Parameter(torch.Tensor(1))
        self.scale = nn.Parameter(torch.Tensor(1, dim))
        self.tanh = nn.Tanh()

        self.reset_parameters()

    def reset_parameters(self):

        weight_initialbd = 0.1
        scale_initialnd = 0.1
        bias_initialbd = 0.1

        self.weight.data.uniform_(-weight_initialbd, weight_initialbd)
        self.scale.data.uniform_(-scale_initialnd, scale_initialnd)
        self.bias.data.uniform_(-bias_initialbd, bias_initialbd)

    def forward(self, z):

        activation = F.linear(z, self.weight, self.bias)

        return z + self.scale * self.tanh(activation)

class flow_log_det_Jacobian(nn.Module):

    def __init__(self, affine):
        super().__init__()

        self.weight = affine.weight
        self.bias = affine.bias
        self.scale = affine.scale
        self.tanh = affine.tanh

    def forward(self, z):
        activation = F.linear(z, self.weight, self.bias)
        psi = (1 - self.tanh(activation) ** 2) * self.weight  # please replace "self.tanh" by "F.tanh"
        det_grad = 1 + torch.mm(psi, self.scale.t())

        return safe_log(det_grad.abs())
    
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

    def forward(self, x):
        return self.map(x), None
  
   
class MLP_shift(nn.Module):
    def __init__(self, params):
        super(MLP_shift, self).__init__()
        self.map = build_model(params).to(params['device'])

    def forward(self, x):
        return x + self.map(x), None
  
  

class ResNet(nn.Module):
    def __init__(self, network_length, input_dimension, hidden_dimension, output_dimension, b=0.1, stepsize=0.5):
        super(ResNet, self).__init__()

        self.linears = nn.ModuleList([nn.Linear(input_dimension, hidden_dimension)])
        self.linears.extend([nn.Linear(hidden_dimension, hidden_dimension) for _ in range(network_length-2)])
        self.linears.extend([nn.Linear(hidden_dimension, output_dimension, bias=False)])
        self.b = b
        self.stepsize = stepsize

    def initialization(self):
        b = self.b
        for l in self.linears:
            # if the length of network is large, then the value of wights should be near 1.0,
            # otherwise the value of derivative of the network will be small (<< 1.0)
            l.weight.data.uniform_(-b, b)
            l.bias.data.uniform_(-b, b)

    def forward(self, z):
        
        ll = self.linears[0]
        x = ll(z)
        
        for l in self.linears[1:-1]:
            print('{}-th layer:'.format(l))
            x = x + self.stepsize * l(x)
            x = torch.tanh(x)  # F.relu(x)

        ll = self.linears[-1]
        x = ll(x)

        return x, None
