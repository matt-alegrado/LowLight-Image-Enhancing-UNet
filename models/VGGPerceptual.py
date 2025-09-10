import torch
import torch.nn as nn
from torchvision import models

class VGGPerceptual(nn.Module):
    def __init__(self, requires_grad=False):
        super().__init__()
        # Load pretrained VGG16
        vgg_pretrained = models.vgg16(weights=models.VGG16_Weights.DEFAULT).features

        # We will extract the features at these layer indices:
        # - relu1_2  → index 3
        # - relu2_2  → index 8
        # - relu3_3  → index 15
        # - relu4_3  → index 22
        # (These indices correspond to layers in torchvision’s vgg16.features)
        self.slice1 = torch.nn.Sequential(*[vgg_pretrained[x] for x in range(0,  4)])   # up to relu1_2
        self.slice2 = torch.nn.Sequential(*[vgg_pretrained[x] for x in range(4,  9)])   # relu2_2
        self.slice3 = torch.nn.Sequential(*[vgg_pretrained[x] for x in range(9, 16)])   # relu3_3
        self.slice4 = torch.nn.Sequential(*[vgg_pretrained[x] for x in range(16, 23)])  # relu4_3

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x):
        """
        Input x should be a 3×H×W tensor (or batch of them),
        normalized the same way that VGG expects: mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225], and range [0,1].
        Returns a list of feature activations from each “slice.”
        """
        h = x
        f1 = self.slice1(h)
        h  = f1
        f2 = self.slice2(h)
        h  = f2
        f3 = self.slice3(h)
        h  = f3
        f4 = self.slice4(h)
        return [f1, f2, f3, f4]
