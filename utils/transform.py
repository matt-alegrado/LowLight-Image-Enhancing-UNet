from torchvision import transforms
import torch
import random

class SlightDimTransform:
    """
    Given a PIL image or a Tensor in [0,1], randomly scale brightness by a factor
    between 0.8 and 1.0. Returns a Tensor in [0,1].
    """

    def __init__(self, base_size=256):
        self.base_size = base_size
        self.to_tensor = transforms.ToTensor()

    def __call__(self, img):
        """
        img: PIL.Image or Tensor in [0,1]
        """
        # 1) Ensure it’s a torch.Tensor in [0,1]
        if not torch.is_tensor(img):
            img = self.to_tensor(img)  # now img is [C×H×W], values ∈ [0,1]

        # 2) Choose a random brightness factor f ∈ [0.8, 1.0]
        f = random.uniform(0.5, .8)

        # 3) Scale
        img_dimmed = img * f

        # 4) Clamp just in case (stay in [0,1])
        return img_dimmed.clamp(0.0, 1.0)