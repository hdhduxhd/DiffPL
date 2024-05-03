import torch
import torch.nn.functional as F

class GaussianDiffusion:
    def __init__(
        self,
        timesteps=2000,
        start=0.0001,
        end=0.02
    ):
        self.timesteps = timesteps
        self.start = start
        self.end = end
        self.step = (end - start) / self.timesteps
        
    
    # forward diffusion (using the nice property): q(x_t | x_0)
    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        batch_size = t.shape[0]
        alphas_cumprod = []
        for i in range(batch_size):
            temp = 1
            print(t[i])
            for minus in range(int(t[i])):
                temp *= (1 - (self.start+(t[i]-minus)*self.step))
            alphas_cumprod.append(temp)
        alphas_cumprod_t = torch.stack(alphas_cumprod,dim=0)
        sqrt_alphas_cumprod_t = torch.sqrt(alphas_cumprod_t).reshape(batch_size, *((1,) * (len(x_start.shape) - 1)))
        sqrt_one_minus_alphas_cumprod_t = torch.sqrt(1.0 - alphas_cumprod_t).reshape(batch_size, *((1,) * (len(x_start.shape) - 1)))

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

from torch.autograd import Variable
import torch.nn.functional as F
def sample_gumbel(shape, eps=1e-20, tens_type=torch.FloatTensor):
    """Sample from Gumbel(0, 1)"""
    U = Variable(tens_type(*shape).uniform_(), requires_grad=False)
    return -torch.log(-torch.log(U + eps) + eps)

# modified for PyTorch from https://github.com/ericjang/gumbel-softmax/blob/master/Categorical%20VAE.ipynb
def gumbel_softmax_sample(logits, temperature):
    """ Draw a sample from the Gumbel-Softmax distribution"""
    y = logits + sample_gumbel(logits.shape, tens_type=type(logits.data)).cuda()
    return F.softmax(y / temperature, dim=1)

def onehot_from_logits(logits, eps=0.0):
    """
    Given batch of logits, return one-hot sample using epsilon greedy strategy
    (based on given epsilon)
    """
    # get best (according to current policy) actions in one-hot form
    argmax_acs = (logits == logits.max(1, keepdim=True)[0]).float()
    if eps == 0.0:
        return argmax_acs

def get_rep_outputs(logits, temperature, hard):    
    y = gumbel_softmax_sample(logits, temperature)
    if hard:
        y_hard = onehot_from_logits(y)
        y = (y_hard - y).detach() + y
    return y
