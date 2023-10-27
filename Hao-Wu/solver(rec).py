import numpy as np
import torch
import torch.nn as nn
import scipy
from scipy.sparse.linalg import LinearOperator

from others import sample_z, copy_net, tensor_to_numpy
from hamiltonian import G_quadratic_loss, dp_evaluate
from evaluation import residual_evaluate, HO_sol, model_evaluation, eval_traj_error

def get_init_p(flow, phi, dim, device='cpu'):
    samples = sample_z(30000, dim, device)
    xs, _ = flow(samples)
    phi_mean = torch.mean(phi(xs))
    p0 = torch.autograd.grad(phi_mean, flow.parameters())
    p0 = tensor_to_numpy(nn.utils.parameters_to_vector(p0))
    return p0



def build_linearsystem_solver(ls_solver):
    exp_af = 'lambda _ : scipy.sparse.linalg.' + ls_solver
    return eval(exp_af)(None)
  
  
  
def linear_system_solver(flow, flow_auxil, pstate, samples, lstr, solver_type='explicit', etaguess=None, print_type=False):
    if solver_type=='explicit':
        xi, res, errrel = explicit_solver_ls(flow, flow_auxil, pstate, samples, lstr)
    else:
        lssolver = build_linearsystem_solver(solver_type)
        xi, res, errrel = iterative_solver_ls(lssolver, flow, flow_auxil, pstate, samples, lstr, etaguess, print_type)
    return xi, res , errrel 



def explicit_solver_ls(flow, flow_auxil, pstate, samples, lstr):
    xs, _ = flow(samples)
    copy_net(flow, flow_auxil)
    xs_auxil, _ = flow_auxil(samples)

    g1 = torch.autograd.grad(xs, flow.parameters(), grad_outputs=xs_auxil, 
                           allow_unused=True, retain_graph=True, create_graph=True)
    vec_g1 = nn.utils.parameters_to_vector(g1)/samples.shape[0]
    n_params = len(vec_g1)

    G = torch.zeros([n_params, n_params]).to(xs.device)
    for i in range(n_params):
        g2 = torch.autograd.grad(vec_g1[i], flow_auxil.parameters(), allow_unused=True, retain_graph=True)
        G[:, i] = nn.utils.parameters_to_vector(g2)
    xi, res1, r1, s1 = np.linalg.lstsq(G, pstate, rcond=lstr)
  
    res2 = np.matmul(G, xi) - pstate
    print('residual:', res1, res2)
    errnorm = np.linalg.norm(res2, 2)
    errrel = errnorm/np.linalg.norm(pstate, 2)
    return xi, res2, errrel

def iterative_solver_ls(lssolver, flow, flow_auxil, pstate, samples, lstr, xiguess=None, print_type=False):
    xs, _ = flow(samples)
    copy_net(flow, flow_auxil)
    xs_auxil, _ = flow_auxil(samples)
    g1 = torch.autograd.grad(xs, flow.parameters(), grad_outputs=xs_auxil, 
                           allow_unused=True, retain_graph=True, create_graph=True)
    if print_type:
        print('g1:',g1)
    vec_g1 = nn.utils.parameters_to_vector(g1)/samples.shape[0]
    n_params = len(vec_g1)

    def mv(v):
        eta = torch.Tensor(v).to(xs.device)
        pre2 = torch.sum(vec_g1 * eta)
        grad_pre2 = torch.autograd.grad(pre2, flow_auxil.parameters(), 
                           allow_unused=True, retain_graph=True, create_graph=True)
        grad_pre2 = nn.utils.parameters_to_vector(grad_pre2)
        return tensor_to_numpy(grad_pre2)
  
    Gop = LinearOperator((n_params, n_params), matvec=mv)
  #print('eta guess:', xiguess)
  #if xiguess!=None:
  #  err = Gop.matvec(xiguess) - pstate
  #  errnorm = np.linalg.norm(err, 2)
  #  errrel = errnorm/np.linalg.norm(pstate, 2)
  #  print('err for guess:', errnorm, errrel)
    
    dalpha, info = lssolver(Gop, pstate, x0=xiguess, tol=lstr, maxiter=None, M=None, callback=None)
    err = Gop.matvec(dalpha) - pstate
    errnorm = np.linalg.norm(err, 2)
    errrel = errnorm/np.linalg.norm(pstate, 2)
    return dalpha, err, errrel



class Implicit_Euler():
    def __init__(self, dt, T):
        super(Implicit_Euler, self).__init__()
        self.dt = dt
        self.T = T

    def step(self, H_system, flow, flow_auxil, pstate, params, lstr, i, etaguess, print_type=False, pe_samplesize=None):
        samples = sample_z(params['nsamples'], params['dim'], params['device'])
        ###d_alpha, d_p, G, s1 = relaxed_Lagrangian_dynamics(H_system, flow, flow_auxil, pstate, samples, lstr)
        tol = lstr
        xi, pstate, res, err = fixed_point_optim_solver(H_system, flow, flow_auxil, pstate, self.dt, samples, tol, 
                       params['px_iters'], params['xi_lr'], params['ls_solver_type'], etaguess, print_type, pe_samplesize)
        alpha = nn.utils.parameters_to_vector(flow.parameters()) + torch.Tensor(self.dt * xi).to(params['device'])
        return alpha, pstate, res, err, xi, samples



def fixed_point_optim_solver(H_system, flow, flow_auxil, pstate, h, samples, lstr, px_iters, lr, solver_type='explicit', eta_guess=None, print_type=None, pe_samplesize=None):
    alpha_k = nn.utils.parameters_to_vector(flow.parameters())
    device = alpha_k.device
    alpha_k = tensor_to_numpy(alpha_k)
  
    dalpha = None
    #dalpha, dp, G, s1 = relaxed_Lagrangian_dynamics(H_system, flow, flow_auxil, pstate, samples, lstr, dalpha, solver='iterative')
    dalpha, res, err = linear_system_solver(flow, flow_auxil, pstate, samples, lstr, solver_type, eta_guess, False)#print_type)
    eta = torch.nn.Parameter(data=torch.Tensor(dalpha).to(device), requires_grad=True)
    eta_optimizer = torch.optim.SGD([eta], lr=lr)
    if print_type:
        print('initial residual')
        print(residual_evaluate(flow, flow_auxil, eta, pstate, samples))
    for i in range(px_iters):
        #print('Start of {}-th inner iteration:'.format(i))
        #dalpha, dp, G, s1 = relaxed_Lagrangian_dynamics(H_system, flow, flow_auxil, pstate, samples, lstr, dalpha, solver='iterative')
        eta_optimizer.zero_grad()
        etaloss, gsum, lterm = G_quadratic_loss(flow, flow_auxil, eta, pstate, samples)
   
        etagrad = torch.autograd.grad(etaloss, eta, allow_unused=True, retain_graph=True, create_graph=True)
        etagrad = nn.utils.parameters_to_vector(etagrad)
    
        etaloss.backward()
        eta_optimizer.step()
        #print('grad of eta loss:', etagrad)

        alpha = torch.Tensor(alpha_k).to(device) + h * eta
        nn.utils.vector_to_parameters(alpha, flow.parameters())
        copy_net(flow, flow_auxil)
        if print_type:
            print('residual in {}-th inner iteration:'.format(i))
            print(residual_evaluate(flow, flow_auxil, eta, pstate, samples))
            print('grad of eta loss:', etagrad)
    if pe_samplesize==None:
        pe_samples = samples
    else:
        pe_samples = sample_z(pe_samplesize, samples.shape[1], device)
    dp = dp_evaluate(H_system, flow, flow_auxil, pstate, eta, pe_samples)
    p = pstate + h * dp
    return tensor_to_numpy(eta), p, res, err

def residual_evaluate(flow, flow_auxil, xi, pstate, samples):
    xs, _ = flow(samples)
    copy_net(flow, flow_auxil)
    xs_auxil, _ = flow_auxil(samples)

    g1 = torch.autograd.grad(xs, flow.parameters(), grad_outputs=xs_auxil, 
                           allow_unused=True, retain_graph=True, create_graph=True)
    vec_g1 = nn.utils.parameters_to_vector(g1)/samples.shape[0]
    n_params = len(vec_g1)
    #print(vec_g1, xi)
    g3 = torch.sum(vec_g1 * xi)
    g4 = torch.autograd.grad(g3, flow_auxil.parameters(), allow_unused=True, retain_graph=True, create_graph=True)
    Gx = nn.utils.parameters_to_vector(g4)
    ### No factor 2 this time, since we are differentiating w.r.t xi
    #kinetic_E = 2. * 0.5 * G_sum
    p = torch.Tensor(pstate).to(xi.device)
    diff = tensor_to_numpy(Gx - p)
    return np.linalg.norm(diff)