import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from models.VAE import VAE
from models.Discriminator import Discriminator
from models.VGGPerceptual import VGGPerceptual
from utils.loss import recon_loss, kld_loss, adv_loss, perceptual_loss, ssim_loss


class UNetGAN(nn.Module):
    def __init__(self, generator, discriminator, recon_weight=1.0, kld_weight=0.01, adv_weight=0.1, perc_weight=0.5, ssim_weight=0.5):
        super().__init__()
        self.u_net = generator
        self.discriminator = discriminator
        # self.vgg = VGGPerceptual()
        self.vgg = None

        self.recon_weight = recon_weight
        self.kld_weight = kld_weight
        self.adv_weight = adv_weight
        self.perc_weight = perc_weight
        self.ssim_weight = ssim_weight

    def forward(self, x):
        return self.u_net(x)

    def generate(self, x):
        return self.u_net(x)

    def gen_loss(self, x_real, x_fake, adv_scaling=1):
        alpha = self.recon_weight
        beta = self.perc_weight
        gamma = self.adv_weight * adv_scaling
        delta = self.ssim_weight

        d_fake = self.discriminator(x_fake)  # Using mean_x as reconstructed image

        # All losses
        recon_loss_score = recon_loss(x_real, x_fake) if alpha > 0 else 0
        adv_loss_score = -d_fake.mean() if gamma > 0 else 0  # Hinge loss for generator
        # perc_loss_score = perceptual_loss(self.vgg, x_real, x_fake) if beta > 0 else 0
        perc_loss_score = 0
        ssim_loss_score = ssim_loss(x_real, x_fake) if delta > 0 else 0

        loss = alpha * recon_loss_score + beta * perc_loss_score + gamma * adv_loss_score + delta * ssim_loss_score
        output = {
            'full': loss,
            'recon': recon_loss_score,
            'adv': adv_loss_score,
            'perc': perc_loss_score,
            'ssim': ssim_loss_score
        }
        return output

    def disc_loss(self, x_real, x_fake):
        # Detach from grad graph for discriminator training
        d_real = self.discriminator(x_real)
        d_fake = self.discriminator(x_fake)

        return adv_loss(d_real, d_fake)  # Hinge loss for discriminator

