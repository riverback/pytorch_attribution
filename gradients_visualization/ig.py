from .base import VanillaGradient

import torch
import numpy as np

class IntegratedGradients(VanillaGradient):
    def get_mask(self, img: torch.Tensor, 
                 target_class: torch.Tensor, 
                 baseline='black', 
                 steps=128, 
                 process=lambda x: x):
        if baseline == 'black':
            baseline = torch.ones_like(img, device=self.device) * torch.min(img).detach()
        elif baseline == 'white':
            baseline = torch.ones_like(img, device=self.device) * torch.max(img).detach()
        else:
            baseline = torch.zeros_like(img, device=self.device)

        B, C, H, W = img.size()
        grad_sum = torch.zeros((B, C, H, W), device=self.device)
        image_diff = img - baseline

        for step, alpha in enumerate(np.linspace(0, 1, steps)):
            image_step = baseline + alpha * image_diff
            grad_sum += process(super(IntegratedGradients,
                                self).get_mask(image_step, target_class))
        return grad_sum * image_diff.detach() / steps