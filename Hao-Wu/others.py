import numpy as np
import torch
import torch.nn as nn

def sample_z(n_samples, dim, device='cpu'): 
    return torch.normal(0., 1., size=[n_samples, dim]).to(device)

def copy_net(net1, net2):
    netparams = nn.utils.parameters_to_vector(net1.parameters())
    device = netparams.device
    netp = tensor_to_numpy(netparams)
    nn.utils.vector_to_parameters(torch.Tensor(netp).to(device), net2.parameters())
    return None

def tensor_to_numpy(z):
    if z.device=="cpu":
        return z.detach().numpy()
    else:
        return z.cpu().detach().numpy()
    
def gmm_sample(mu_set, var_set, nsamples_set, dim, device='cpu'):
    """
    :param mu: torch.Tensor (features)
    :param var: torch.Tensor (features) (note: zero covariance)
    :return: torch.Tensor (nb_samples, features)
    """
    out = []
    for i in range(len(mu_set)):
        w = torch.randn([nsamples_set[i], dim])
        ws = torch.matmul(w, var_set[i]) + mu_set[i]
        out += ws
    z = torch.stack(out, dim=0)
#         out += [
#             torch.zeros(nsamples_set[i], dim).normal_(mean=mu_set[i], std=var_set[i])
#         ]
    
    return z.to(device)

class init_sampler(object):
    def __init__(self, muset=None, varset=None):
        self.muset = muset
        self.varset = varset
        
    def sampler(self, n_samples, dim, device='cpu'):
        if self.muset==None:
            return sample_z(n_samples, dim, device)
        else:
            return gmm_sample(self.mu_set, self.var_set, nsamples_set, dim, device)

# create list of m numbers [n, 2n, 3n,...,mn] for plotting loss curves
def create_nodes(m, n):
    nodes = torch.ones(m, 1)
    for i in range(1, m + 1):
        nodes[i-1] = i * n
    return nodes


# create nodes for plotting the graph of psi
def psi_nodes(interval_width, num, dim):

    NUM = (num+1) * (num+1)

    torchv = torch.ones(NUM, dim)

    stepsize = 2 * interval_width / num

    for k in range(1, num+2):
        for l in range(1, num+2):
            torchv[(k-1)*(num+1)+(l-1)][0] = (k-1)*stepsize-interval_width
            torchv[(k-1)*(num+1)+(l-1)][1] = (l-1)*stepsize-interval_width

    return torchv