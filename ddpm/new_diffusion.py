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
            for minus in range(int(t[i])):
                temp *= (1 - (self.start+(t[i]-minus)*self.step))
            alphas_cumprod.append(temp)
        alphas_cumprod_t = torch.stack(alphas_cumprod,dim=0)
        sqrt_alphas_cumprod_t = torch.sqrt(alphas_cumprod_t).reshape(batch_size, *((1,) * (len(x_start.shape) - 1)))
        sqrt_one_minus_alphas_cumprod_t = torch.sqrt(1.0 - alphas_cumprod_t).reshape(batch_size, *((1,) * (len(x_start.shape) - 1)))

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
