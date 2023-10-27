import numpy as np
import torch
import matplotlib.pyplot as plt
from hamiltonian import sample_z

def plot_traj(test_set, dt, T):
  pset = torch.stack(test_set, dim=0).detach().numpy()
  timeset = np.linspace(0, T, num = pset.shape[0])
  print('shape:', pset.shape, timeset.shape)
  for i in range(pset.shape[1]):
    point_traj = pset[:, i, 0]
    plt.plot(timeset, point_traj, label='{}-th point'.format(i))

  plt.xlabel('x - axis')
  plt.ylabel('y - axis')
  plt.show()
  return None

def plot_hist(model, true_rho, t=0., n_samples=1000, dim=1, density_type=None):
  z = sample_z(n_samples, dim, density_type)
  x, _ = model(z)
  x = x.detach().numpy()
  xt = np.sort(x.reshape(-1))
  true_y = true_rho(xt, t)
  print('min and max for samples:', np.min(x), np.max(x))
  plt.plot(xt, true_y)
  _ = plt.hist(x, bins=100, density=True)  # arguments are passed to np.histogram
  #plt.title("Histogram with 'auto' bins")
  plt.show()
  return None

def plot_phi(model, true_phi, t, xlim=5.0, step=0.001):
  xs = np.arange(- xlim, xlim, step)
  x = torch.Tensor(xs).reshape(-1, 1)
  y = model(x)
  y = y.detach().numpy()
  y0 = torch.Tensor([[0.]])
  phi0 = model(y0).detach().numpy()
  true_y = true_phi(x, t).detach().numpy()
  plt.plot(xs, y - phi0)#, color='green', marker='o', linestyle='dashed', linewidth=2, markersize=12)
  plt.plot(xs, true_y)
  plt.show()
  
def log_density(z, model, params):
  model_x, log_j = model(z)
  log_const = - np.log(2*np.pi) /2.
  return log_const - z**2/2. + log_j, model_x

def compute_density(z, model, params):
  log_d, model_x = log_density(z, model, params)
  return torch.exp(log_d), model_x