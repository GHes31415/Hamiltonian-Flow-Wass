import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

from flow_and_mlp import NormalizingFlow, MLP_shift
from solver import get_init_p
from hamiltonian import Hamiltonian_System, quadratic_init
from hamiltonian import quadratic_potential, cos_potential, Coulomb_potential, comb_potential, entropy_potential, interaction_potential #fourth_potential
from evaluation import HO_sol

from others import sample_z, copy_net, tensor_to_numpy

def create_eval_rec(params, dt, niter):
    points = np.linspace(-5., 5., 100*params['dim'])
    test_points = sample_z(params['plot_test_size'], params['dim']).to(params['device'])

    KE_rec = np.zeros(niter)
    PE_rec = np.zeros(niter)
    H_rec = np.zeros(niter)
    traj_err_rec = np.zeros(niter)
    traj_rec = list() #np.zeros([1 + np.math.floor(niter/params['nrec']), test_points.shape[0], params['dim']])
    #traj_rec[0] = tensor_to_numpy(test_points)
    return test_points, traj_rec, KE_rec, PE_rec, H_rec, traj_err_rec

def create_param_rec(niter, nrec, n_params):
    #len_rec = 1 + np.math.floor(niter/nrec)
    theta_rec = list() #np.zeros([len_rec, n_params])
    p_rec = list() #np.zeros([len_rec, n_params])
    errrel_rec = np.zeros(niter)
    return theta_rec, p_rec, errrel_rec

def variable_init(params, init_func, niter, dt, T):
    if params['net_type']=='mlp':
        flow = MLP_shift(params).to(params['device'])
        flow_auxil = MLP_shift(params).to(params['device'])
    elif params['net_type']=='flow':
        flow = NormalizingFlow(params['dim'], params['flow_length']).to(params['device'])
        flow_auxil = NormalizingFlow(params['dim'], params['flow_length']).to(params['device'])
    pstate = get_init_p(flow, init_func, params['dim'], params['device'])
    eta = None
    #PreTrain(flow, params, 1500, 0.003, Gauss_density)
    return flow, flow_auxil, pstate, eta

def HS_init(params):
    if params['potential_type']=='quadratic':
        potential_func = quadratic_potential(params['potential_coef']).potential_eval
    elif params['potential_type']=='cos':
        potential_func = cos_potential(params['potential_coef'], params['potential_coef2']).potential_eval
#     elif params['potential_type']=='fourth':
#         potential_func = fourth_potential(params['potential_coef']).potential_eval
    elif params['potential_type']=='interaction':
        potential_func = interaction_potential(params['potential_coef']).potential_eval
    elif params['potential_type']=='Coulomb':
        potential_func = Coulomb_potential(params['potential_coef']).potential_eval
    elif params['potential_type']=='entropy':
        potential_func = entropy_potential(params['potential_coef']).potential_eval
    elif params['potential_type']=='combine':
        potential_func = comb_potential(params['qw'], params['iw'], params['potential_quad'], params['potential_interact']).potential_eval
    phi_init = quadratic_init(params['phiweight'], params["phi_pos"]).func
    H_system = Hamiltonian_System(params['dim'], potential_func, phi_init)
    
    ho_sol = HO_sol(params, H_system)
    
    return ho_sol, H_system

def plot_rec(KE_rec, PE_rec, H_rec):
    plt.plot(H_rec)
    plt.show()
    plt.plot(KE_rec)
    plt.show()
    plt.plot(PE_rec)
    plt.show() 
  
  

def create_rec(params, dt, T):
    niter = int(T/dt)
    points = np.linspace(-5., 5., 100*params['dim'])
    test_points = torch.Tensor(points).reshape(-1, params['dim']).to(params['device'])
    points_list= list()
    points_list.append(test_points)

    KE_rec = np.zeros(niter)
    PE_rec = np.zeros(niter)
    H_rec = np.zeros(niter)
    traj_err_rec = np.zeros(niter)
    traj_rec = np.zeros([niter + 1, test_points.shape[0], params['dim']])
    traj_rec[0] = tensor_to_numpy(test_points)
    return niter, test_points, traj_rec, KE_rec, PE_rec, H_rec, traj_err_rec