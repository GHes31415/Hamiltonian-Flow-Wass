import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from scipy.sparse.linalg import LinearOperator
import os

from plot_func import plot_hist, plot_phi

from others import copy_net, tensor_to_numpy


class HO_sol(object):
    def __init__(self, params, h_system):
        self.params = params
        self.h_system = h_system
        if params['potential_type']=='quadratic':
            self.c = np.diag(tensor_to_numpy(params['potential_coef']))
            self.b = (np.diag(tensor_to_numpy(params['phiweight'])) **2) / self.c
            self.phase0 = - np.arctan(self.b)
            self.coef = np.sqrt(1 + self.b ** 2)
        return None

    ### def mu_t, Sigma_t, Gauss_param
    def mu_t(self, t):
        return np.array([[0. for i in range(self.params['dim'])]])

    def Sigma_t(self, t):
        sig = self.coef * np.cos(self.c * t - self.phase0)
        return np.diag(sig)#coef * np.identity(self.params['dim'])
  
    def Gauss_param(self, t):
        return self.mu_t(t), self.Sigma_t(t)
    
    def true_y(self, xset, t):
        sig = self.Sigma_t(t)
        y = xset * sig
        return y
    
    def euler_y(self, xset, dt, nstep):
        return self.h_system.numerical_solve(xset)

    def traj(self, x0, Tset):
        pass

    def traj_proj(self, x0, Tset, i=0, j=1):
        ci, cj = self.c[i], self.c[j]
        thetai, thetaj = self.phase0[i], self.phase0[j]
        sigi = self.coef[i] * np.cos(ci * Tset - thetai)
        sigj = self.coef[j] * np.cos(cj * Tset - thetaj)
        xi, xj = x0[i] * sigi, x0[j] * sigj
        return xi, xj

    def plot_traj_proj(self, x0, Tset, i=0, j=1):
        xi, xj = self.traj_proj(x0, Tset, i, j)
        plt.plot(xi, xj)
        plt.show()
    
def proj_Gaussian(x, mu, Sigma, vec):
    print(mu)
    xproj = np.matmul(x, vec.T)
    muproj = np.sum(vec * mu)
    sigmaproj = np.matmul(vec, np.matmul(Sigma, vec.T)).reshape(1)
    print('proj of sigma:', sigmaproj)
    xscaled = (xproj - muproj)/sigmaproj
    f = np.exp( - xscaled ** 2 / 2.) / (sigmaproj * np.sqrt(2 * np.pi))
    return f, xproj



def model_evaluation(flow, ho_sol, params, sample, plotsample, t):
    mu, Sigma = ho_sol.Gauss_param(t)

    def sol_func(x):
        if Sigma[0, 0]==0.:
            return xshift
        print(x, mu, t)
        premu = torch.Tensor(mu).reshape(1, -1)
        xshift = x - premu.to(x.device) * t
        xshift = tensor_to_numpy(xshift)
        return np.matmul(xshift, Sigma)
  
    traj_err = eval_traj_error(sample, flow, sol_func)
    plot_projection(flow, params, plotsample, mu, Sigma, t, num=100)
    return traj_err



def eval_traj_error(sample, flow, true_func):
    true_sol = true_func(sample)
    xset, _ = flow(sample)
    diff = true_sol - tensor_to_numpy(xset)
    err = np.linalg.norm(diff, axis=1)
    return np.mean(err)

font = {'family': 'serif',
        'color':  'darkred',
        'weight': 'normal',
        'size': 16,
        }

def plot_projection(save_path, flow, params, plotsample, mu, Sigma, i, num=100, t=None):
    xt, _ = flow(plotsample)
    x = tensor_to_numpy(xt)

    for d in range(2):#params['dim']):
        vec = np.zeros([1, params['dim']])
        vec[0, d] = 1.0
        xproj = np.matmul(x, vec.T)
#         print('histogram plot of projection onto the {}-th dimension at time t={}:'.format(i, t))
#         print('min and max for samples projection:', np.min(xproj), np.max(xproj))
        testx = np.zeros([num, params['dim']])
        testx[:, d] = np.linspace(-4, 4, num=num)
        
        fproj, txproj = proj_Gaussian(testx, mu, Sigma, vec)
        plt.plot(txproj, np.abs(fproj), label='True density')
        plt.hist(xproj, bins=100, density=True, label='Sample histogram')
        plt.xlim([-4., 4.])
        plt.ylim([0., 1.])
        if t!=None:
            plt.text(-3, 0.65, 'T={0:.2f}'.format(t), fontdict=font)
        plt.legend(loc='upper left')
        save_name = os.path.join(save_path, "(" + params['potential_type'] + ") " + "sample histogram {}-dim {}".format(d, i))
        #plt.title("sample histogram in {}-th dimension ".format(1+d))
        plt.savefig(save_name)
        #_ = plt.hist(xproj, bins=100, density=True)  # arguments are passed to np.histogram
        #plt.title("Histogram with 'auto' bins")
        plt.show()
    return None



def residual_evaluate(flow, flow_auxil, dalpha, pstate, samples):
    xs, _ = flow(samples)
    copy_net(flow, flow_auxil)
    xs_auxil, _ = flow_auxil(samples)

    g1 = torch.autograd.grad(xs, flow.parameters(), grad_outputs=xs_auxil, 
                           allow_unused=True, retain_graph=True, create_graph=True)
    vec_g1 = nn.utils.parameters_to_vector(g1)/samples.shape[0]
    n_params = len(vec_g1)

    def mv(v):
        g1 = torch.autograd.grad(xs, flow.parameters(), grad_outputs=xs_auxil, 
                           allow_unused=True, retain_graph=True, create_graph=True)
        vec_g1 = nn.utils.parameters_to_vector(g1)/samples.shape[0]
        xi = torch.Tensor(v).to(device)
        pre2 = torch.sum(vec_g1 * xi)
        grad_pre2 = torch.autograd.grad(pre2, flow_auxil.parameters(), 
                           allow_unused=True, retain_graph=True, create_graph=True)
        grad_pre2 = nn.utils.parameters_to_vector(grad_pre2)
        return tensor_to_numpy(grad_pre2)
  
    Gop = LinearOperator((n_params, n_params), matvec=mv)

    err = Gop.matvec(dalpha) - pstate
    errnorm = np.linalg.norm(err, 2)
    errrel = errnorm/np.linalg.norm(pstate, 2)
    print('err:', errnorm, errrel)
    return None


def estimate_velocity_position(flow, f_auxil, test_samples, dalpha, dt):
    test_samples.requires_grad = True
    xs, _ = flow(test_samples)
    real_velocity = torch.zeros_like(xs).to(xs.device)
    est_velocity = torch.zeros_like(xs).to(xs.device)
    new_alpha = nn.utils.parameters_to_vector(flow.parameters()) + dt * dalpha
    nn.utils.vector_to_parameters(new_alpha, f_auxil.parameters())
    xs_auxil, _ = f_auxil(test_samples)
    dtda_rec = torch.zeros(xs.shape[0], len(nn.utils.parameters_to_vector(flow.parameters())))
    print(flow, f_auxil)
    for i in range(xs.shape[0]):
    #print('i:', i)
    #print(xs_auxil)
    #print(xs)
        dTda_i = torch.autograd.grad(xs[i, 0], flow.parameters(), allow_unused=True, retain_graph=True, create_graph=True)
        vec_dTda = nn.utils.parameters_to_vector(dTda_i)
        real_velocity[i, 0] = torch.sum(vec_dTda * dalpha)
        dtda_rec[i, :] = vec_dTda
        dx_i = xs_auxil[i, 0] - xs[i, 0]
        est_velocity[i, 0] = dx_i/dt 
    #print('dtda:', dtda_rec)
    #print('dalpha:', dalpha)

    return real_velocity, est_velocity, xs



