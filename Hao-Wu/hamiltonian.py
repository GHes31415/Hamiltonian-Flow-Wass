import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from others import sample_z, copy_net, tensor_to_numpy



class Hamiltonian_System(object):
    def __init__(self, dim, potential, phi_zero=None, true_map=None, true_density_proj=None,device='cpu'):
        self.device = device
        self.dim = dim
        self.potential = potential
        self.phi_zero = phi_zero
        self.true_map = true_map
        self.true_density_proj = true_density_proj
       
    def potential_eval(self, xs, log_jac=None):
        if self.potential==None:
            print('Warning: zero potential')
            return 0
        else:
            return self.potential(xs, log_jac)
    
    def Hamiltonian_eval(self, flow, flow_auxil, eta, samples):
        '''
        Evaluation of the Hamiltonian, it returns the kinetic energy and the potential energy
        '''
        zero_vec = torch.zeros_like(eta).to(self.device)
        '''Evaluation of kinetic energy'''
        _, KE, _ = G_quadratic_loss(flow, flow_auxil, eta, zero_vec, samples)
        xs, log_jac = flow(samples)
        '''Evaluation of potential energy'''
        PE = self.potential_eval(xs, log_jac)
        return KE, PE
        
    def true_sol(self, xs):
        pass
        
    def grad_potential(self, xs, cgraph=True):
        #xs = torch.Tensor(x).to(self.device)
        xs.requires_grad=True
        pe = self.potential(xs)
        grad_pe = torch.autograd.grad(pe, xs,
                           allow_unused=True, retain_graph=False, create_graph=cgraph)[0]
        return grad_pe
    
    def numerical_step(self, x, v, dt, cgraph=False):
        grad_pe = x.shape[0] * self.grad_potential(x, cgraph)
        xt = x + dt * v
        vt = v - dt * grad_pe
        return xt, vt
    
    def numerical_init(self, xs, phi_init=None):
        if phi_init==None:
            phi_init=self.phi_zero
        xs.requires_grad=True
        # Sum before taking gradient
        pe = torch.sum(phi_init(xs))
        grad_phi = torch.autograd.grad(pe, xs,
                           allow_unused=True, retain_graph=False, create_graph=True)[0]
        v = grad_phi
        return v
    
    def numerical_sol(self, xs, dt, nstep, phi_init=None):
        v = self.numerical_init(xs, phi_init)
        xtraj = np.zeros([1+np.math.floor(nstep/10), xs.shape[0], xs.shape[1]])
        xtraj[0] = tensor_to_numpy(xs)
        vtraj = np.zeros([1+np.math.floor(nstep/10), xs.shape[0], xs.shape[1]])
        vtraj[0] = tensor_to_numpy(v)
        h_rec, pe_rec, ke_rec = np.zeros(nstep), np.zeros(nstep), np.zeros(nstep)
        for i in range(nstep):
            xs = xs.detach_().requires_grad_()#torch.Tensor(tensor_to_numpy(xs)).to(v.device)
            xs, v = self.numerical_step(xs, v, dt)
            
            h_rec[i], pe_rec[i], ke_rec[i] = self.energy_eval(xs, v)
            if (1+i)%10==0:
                #print(i)
                xtraj[int((1+i)/10)] = tensor_to_numpy(xs)
                vtraj[int((1+i)/10)] = tensor_to_numpy(v)
        return xtraj, vtraj, h_rec, pe_rec, ke_rec
    
    def energy_eval(self, xs, vs, log_jac=None):
        ke = np.sum(tensor_to_numpy(vs)**2)/(2. * vs.shape[0])
        pe = self.potential_eval(xs, log_jac)
        pe = tensor_to_numpy(pe)
        energy = ke + pe
        return energy, pe, ke
    
    def estimate_density_proj(self, x, phi_zero):
        pass
    
    def init_velocity(self, x):
        # based on sample set
        # for linear potential, sample set does'nt change the value
        # for interaction potential, depends on the interaction/density
        pass
    
class Coulomb_potential(object):
    def __init__(self, eps):
        self.eps = eps
        
    def potential_eval(self, x, log_jac=None):
        nsamples = x.shape[0]
        y = x #y = torch.Tensor(tensor_to_numpy(x)).to(x.device)
        xsq = torch.sum(x**2, dim=1).reshape(-1, 1)
        ysq = torch.sum(y**2, dim=1).reshape(-1, 1)
        xy = torch.matmul(x, y.T)
        matrix = xsq + (ysq.T) - 2*xy
        #matrix = torch.cdist(x, x, p=2) **2
        matrix = matrix - torch.diag_embed(torch.diagonal(matrix))
        eps_sq = torch.Tensor([self.eps ** 2]).to(x.device)
        
        trunc_matrix = torch.where(matrix>eps_sq, matrix, eps_sq)
        np_m = tensor_to_numpy(matrix)
        #nz = np.sum(np.where(np_m>self.eps**2, np.ones_like(np_m), np.zeros_like(np_m)))
        nz = torch.sum(torch.where(matrix>eps_sq, torch.ones_like(trunc_matrix), torch.zeros_like(trunc_matrix)))
        #print(nz)
        trunc_matrix.pow_(-1)
        re_matrix =torch.where(matrix>eps_sq, trunc_matrix, torch.zeros_like(trunc_matrix))
        re_potential = torch.sum(re_matrix)/nz
        return re_potential

class quadratic_potential(object):
    def __init__(self, potentialweight):
        self.potentialweight = potentialweight
  
    def potential_eval(self, x, log_jac=None):
        y = torch.matmul(x, self.potentialweight)
        return torch.sum(y**2)/(2. * x.shape[0])

class cos_potential(object):
    def __init__(self, potential_coef, potential_coef2):
        self.potential_coef = potential_coef
        self.potential_coef2 = potential_coef2
  
    def potential_eval(self, x, log_jac=None):
        y = torch.matmul(x, self.potential_coef)
        return self.potential_coef2 * torch.sum(torch.cos(y))/(2. * x.shape[0])
    
class entropy_potential(object):
    def __init__(self, potential_coef):
        self.potential_coef = potential_coef
        self.ref_log_density = None
        
    def potential_eval(self, x, log_jac):
        mean_log_jac = log_jac.mean()
        return self.potential_coef * mean_log_jac    
    
class comb_potential(object):
    def __init__(self, qw, iw, pweight, pa=1.):
        self.pweight = pweight
        self.pa = pa
        self.qw = qw
        self.iw = iw
        self.qp = quadratic_potential(pweight).potential_eval
        self.ip = interaction_potential(pa).potential_eval
        
    def potential_eval(self, x, log_jac=None):
        return self.qw * self.qp(x) + self.iw * self.ip(x)
    
class quadratic_init(object):
    def __init__(self, phiweight, pos=1.):
        self.phiweight = phiweight
        self.pos = pos
        
    def func(self, x):
        y = torch.matmul(x, self.phiweight)
        print("positive?", self.pos)
        quad = self.pos * torch.sum(y**2/2., dim=1, keepdim=True) # + torch.matmul(y, self.c)
        print("Phi initial:", quad)
        return quad
    
    
    
def dp_evaluate(H_system, flow, flow_auxil, pstate, eta, samples, samples_log_rho, z_log_rho_func, naive=False):
    xs, log_jac = flow(samples, samples_log_rho)
    copy_net(flow, flow_auxil)
    xs_auxil, _ = flow_auxil(samples, samples_log_rho)

    g1 = torch.autograd.grad(xs, flow.parameters(), grad_outputs=xs_auxil, 
                           allow_unused=True, retain_graph=True, create_graph=True)
    vec_g1 = nn.utils.parameters_to_vector(g1)/samples.shape[0]
    n_params = len(vec_g1)

    g3 = torch.sum(vec_g1 * eta)
    g4 = torch.autograd.grad(g3, flow_auxil.parameters(), allow_unused=True, retain_graph=True, create_graph=True)
    G_sum = torch.sum(eta * nn.utils.parameters_to_vector(g4))
    kinetic_E = 2. * 0.5 * G_sum 
    potential_E = H_system.potential_eval(flow, samples, samples_log_rho, 0.001)
    #print('log det in dp evaluation:', log_jac)
    T1 = torch.autograd.grad(kinetic_E, flow.parameters(), allow_unused=True, retain_graph=True)
    T2 = torch.autograd.grad(potential_E, flow.parameters())
 
    if naive:
        return - tensor_to_numpy(nn.utils.parameters_to_vector(T2))
    
    if T1[0]==None:
        dp1 = np.zeros_like(tensor_to_numpy(nn.utils.parameters_to_vector(T2)))
    else:
        dp1 = tensor_to_numpy(nn.utils.parameters_to_vector(T1))
    dp2 = tensor_to_numpy(nn.utils.parameters_to_vector(T2))
#     print("value of dp1:", dp1)
    print("value of potential energy:", potential_E)
#     print("value of dp2:", dp2)
    dp = dp1 - dp2
    return dp



def G_quadratic_loss(flow, flow_auxil, xi, pstate, samples):
    '''
    flow: weights of the flow
    flow_auxil: unattached copy of the weights of the flow
    xi: 
    
    '''
    xs, _ = flow(samples)
    copy_net(flow, flow_auxil)
    xs_auxil, _ = flow_auxil(samples)

    g1 = torch.autograd.grad(xs, flow.parameters(), grad_outputs=xs_auxil, 
                           allow_unused=True, retain_graph=True, create_graph=True)
    vec_g1 = nn.utils.parameters_to_vector(g1)/samples.shape[0]
    n_params = len(vec_g1)

    g3 = torch.sum(vec_g1 * xi)
    g4 = torch.autograd.grad(g3, flow_auxil.parameters(), allow_unused=True, retain_graph=True, create_graph=True)
    G_sum = torch.sum(xi * nn.utils.parameters_to_vector(g4))
    ### No factor 2 this time, since we are differentiating w.r.t xi
    #kinetic_E = 2. * 0.5 * G_sum
    p = torch.Tensor(pstate).to(xi.device)
    linear_term = torch.sum(p * xi)
    quad_loss = 0.5 * G_sum - linear_term
    return quad_loss, 0.5 * G_sum, linear_term


# class fourth_potential(object):
#     def __init__(self, potentialweight):
#         self.potentialweight = potentialweight
  
#     def potential_eval(self, x, log_jac=None):
#         y = torch.matmul(x, self.potentialweight)
#         return torch.sum(y**4)/(2. * x.shape[0])
    
class interaction_potential(object):
    def __init__(self, a):
        self.a = a
        
    ### deep copy x to get y
    ### keep the value, but detach from the computational graph
    def potential_eval(self, x, log_jac=None):
        nsamples = x.shape[0]
        y = x #y = torch.Tensor(tensor_to_numpy(x)).to(x.device)
        xsq = torch.sum(x**2, dim=1).reshape(-1, 1)
        ysq = torch.sum(y**2, dim=1).reshape(-1, 1)
        xy = torch.matmul(x, y.T)
        matrix = self.a + xsq + (ysq.T) - 2*xy

        matrix.pow_(-1)
        potential = torch.mean(matrix)
        return potential#, matrix
    
#     def numpy_grad_potential(self, xs):
#         nsamples = xs.shape[0]
#         xnorm = np.linalg.norm(xs, axis=1).reshape(-1, 1)
#         xn = xnorm.repeat(nsamples, 1)
#         yn = xn.T
#         xy = np.matmul(xs, xs.T)
#         matrix = self.a + xn**2 + yn**2 - 2*xy
#         denom = matrix**(-2)
#         coef1 = np.sum(denom, axis=1).reshape(-1, 1)
#         t1 = coef1 * xs
#         t2 = np.zeros_like(xs)
#         for i in range(xs.shape[1]):
#             yi = xs[:, i].reshape(1, -1)
#             yi = yi.repeat(nsamples, axis=0)
#             t2[:, i] = np.sum(yi * denom, axis=1)
#         dW = - 2. * (t1 - t2)/(nsamples**2)
#         return dW