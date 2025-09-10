import torch
from loss import *

def train(*args, generator, discriminator, optimizer, train_data, val_data, epochs=10, device='cpu'):
    # TODO logging tensorboard for validation and training loss
    alpha = args[0]
    beta = args[1]
    gamma = args[2]

    generator.to(device)
    generator.train()

    discriminator.to(device)
    discriminator.train()

    for epoch in epochs:
        for t, (x,y) in enumerate(train_data):
            gen, _, mean, var = generator(x)
            D_real = discriminator(x)
            D_gen = discriminator(gen)

            # TODO GaussianNLLLoss
            loss = alpha * perceptual_loss(x, mean, var) + beta * kld_loss(mean, var) + gamma * adv_loss(D_real, D_gen)

            loss.backward()
            optimizer.step()

        # TODO validation



