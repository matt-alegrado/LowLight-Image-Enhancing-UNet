import torch
import numpy as np
from torch.nn import functional as F
import torch.nn as nn
# from torchmetrics.image import MultiScaleStructuralSimilarityIndexMeasure
from torchmetrics.image import StructuralSimilarityIndexMeasure

def recon_loss(batch, mean):
    return F.l1_loss(mean, batch, reduction='mean')

def perceptual_loss(vgg_extractor, real, fake):
    """
    vgg_extractor: an instance of VGGPerceptual()
    fake, real:   [B, 3, H, W] tensors in range [-1, +1] or [0,1] (adjust accordingly)
    Returns: a scalar L1 loss computed over the chosen feature maps.
    """
    # # Normalize both to VGG’s expected input
    # mean = torch.tensor([0.485, 0.456, 0.406], device=fake.device).view(1, 3, 1, 1)
    # std = torch.tensor([0.229, 0.224, 0.225], device=fake.device).view(1, 3, 1, 1)
    #
    # fake_norm = (fake - mean) / std
    # real_norm = (real - mean) / std
    #
    # # Extract feature maps at each slice
    # feats_fake = vgg_extractor(fake_norm)
    # feats_real = vgg_extractor(real_norm)
    #
    # # Compute L1 (or L2) difference at each scale
    # loss = 0.0
    # weights = [1.0, 1.0, 1.0, 1.0]  # you can down‐weight shallow layers if you like
    # for w, f_f, f_r in zip(weights, feats_fake, feats_real):
    #     loss += w * F.l1_loss(f_f, f_r)
    # return loss
    return 0

def adv_loss(D_real, D_gen):
    """
    Hinge loss for adversarial training.
    
    :param D_real: Values from discriminator on real images
    :param D_gen: Values from discriminator on generated images
    :return: Adversarial loss
    """

    real_score = F.relu(1 - D_real)
    gen_score = F.relu(1 + D_gen)
    return (real_score.mean() + gen_score.mean()) / 2


def kld_loss(mean, var, latent_dim):
    loss = torch.mean(-0.5/latent_dim * torch.sum(1 + var - mean ** 2 - var.exp(), dim=1), dim=0)
    if torch.isnan(loss) or torch.isinf(loss):
        raise ValueError("Loss became NaN")
    return loss


def ssim_loss(real, fake):
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(fake.device)
    return 1 - ssim(fake, real)