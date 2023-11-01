from torch.distributions.multivariate_normal import MultivariateNormal
import torchdyn
from torchdyn.core import NeuralODE
from torchdyn.datasets import generate_moons

from torchcfm.conditional_flow_matching import *
from torchcfm.models.models import *
from torchcfm.utils import *



def ref_dist(dim = 2):
    
    return MultivariateNormal(torch.zeros(dim), torch.eye(dim))



def train_cmf(data= torch.tensor, dim= int, batch_size= int,   ) :

    return




