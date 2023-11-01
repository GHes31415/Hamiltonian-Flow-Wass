import torch

def set_params(potential_type, dim=2, device=None):
    '''
    Set the parameters.
    First, set the device.
    Reference distribution is Gaussian.
    Set potential type
    dimension of the problem
    '''
    params = dict()
    if device:
        params['device'] = device
    else:
        params['device'] = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    params['ref_d'] = 'Gaussian'
    
    params['potential_type'] = potential_type
    params['dim'] = dim
    if potential_type in {'quadratic', 'cos', 'Coulomb', 'interaction'}:
        params['net_type'] = 'mlp'
        params['init_b'] = 0.2
        if potential_type in{'Coulomb', 'interaction'}:
            params['pe_samplesize'] = 10000
        else:
            params['pe_samplesize'] = 50000
    else:
        params['net_type'] = 'flow'
        params['pe_samplesize'] = 10000
    
    params['nshows']= 50

    params['nsamples'] = 50000
    
    params['ntestsamples'] = 10000
    params['samesample'] = True
    
    params['plot_test_size'] = 100

    params['nrec'] = 10
    params['n_savefig'] = 50
    params['print_gradient'] = False
    params['print_type'] = False

    params['ls_solver_type'] = 'minres'
    params['restart'] = 2
    params['px_iters'] = 1
    params['xi_lr'] = 0.01
    
    params['p_coef'] = 1.0
    params['c'] = torch.zeros(dim).to(params['device'])
    params['phiweight'] = torch.eye(dim).to(params['device'])
    # Modification 
    params['phi_pos'] = torch.tensor([1.0]).to(params['device'])
    return params