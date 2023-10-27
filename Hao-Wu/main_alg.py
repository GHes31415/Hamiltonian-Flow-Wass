import matplotlib.pyplot as plt
import numpy as np

from others import sample_z, tensor_to_numpy
from flow_and_mlp import NormalizingFlow, MLP_shift
from solver import Implicit_Euler
from hamiltonian import Hamiltonian_system, Quad_Gaussian_pair, quad_potential

def HF_main(params, dt, T):
    QP = quad_potential(params['potentialweight'])
    H_system = Hamiltonian_system(params['dim'], QP.potential_eval)
#     Gauss_quad_geodesic_system = Quad_Gaussian_pair(params)
#     quad_func = Gauss_quad_geodesic_system.true_phi
#     Gauss_density = Gauss_quad_geodesic_system.true_rho

    flow = MLP_shift(params).to(params['device'])
    flow_auxil = MLP_shift(params).to(params['device'])

    #PreTrain(flow, params, 1500, 0.003, Gauss_density)

    #solver = Implicit_Euler(dt, T)
    reca, recb, KE_rec, PE_rec, H_rec, recc, G_rec, p_rec = solver.solve_HS(H_system, flow, flow_auxil, Gauss_density, quad_func,
                                         params, stepsize=0.5,lstr=2e-3)
    w = np.zeros([len(reca), len(reca[0]), params['dim']])
    for i in range(len(reca)):
        w[i] = reca[i]

    plt.plot(H_rec)
    plt.show()
    plt.plot(KE_rec)
    plt.show()
    plt.plot(PE_rec)
    plt.show()  
    return flow, w, recb, KE_rec, PE_rec, H_rec, recc, G_rec, p_rec