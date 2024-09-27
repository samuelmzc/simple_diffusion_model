import numpy as np
import matplotlib.pyplot as plt
import torch as torch
import torchvision as tvis
import torch.nn.functional as F

def variance_schedule(timesteps):
    """
    Implements the beta scheduler lineally with beta1 = 10**-4 and betaT = 0.02 (paper)

    Arguments:
    timesteps -- number of diffusion steps, int

    Returns:
    beta_t -- beta schedule, torch.Tensor
    """

    start = 10**(-4)
    end = 0.02
    return torch.linspace(10**(-4), 0.02, timesteps)


def cumm_alphas(beta_t):
    """
    Compute the cummulative alphas (alpha_t = 1 - beta_t)

    Arguments:
    beta_t -- torch.Tensor

    Returns:
    cumm_alphas -- torch.Tensor
    """

    alpha_t = 1 - beta_t
    return torch.cumprod(alpha_t, dim = 0)



def forward_diffusion_step(image, t, alphacumm_t):
    noise = torch.randn(image.shape)
    return torch.sqrt(alphacumm_t)*image + np.sqrt(1 - alphacumm_t)*noise, noise



    
