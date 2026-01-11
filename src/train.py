import torch
import torch.nn as nn

def vae_loss(recon, x, mu, logvar):
    recon_loss = nn.MSELoss()(recon, x)
    kl = -0.5 * torch.mean(1 + logvar - mu**2 - torch.exp(logvar))
    return recon_loss + kl