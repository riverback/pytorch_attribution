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
    
    def get_smoothed_mask(self, img: torch.Tensor, 
                          target_class: torch.Tensor, 
                          baseline='black', 
                          steps=128, 
                          process_ig=lambda x: x, # used in self.get_mask
                          samples=25,
                          std=0.15,
                          process=lambda x: x**2):
        std = std * (torch.max(img) - torch.min(img)).detach().cpu().numpy()
        
        B, C, H, W = img.size()
        grad_sum = torch.zeros((B, C, H, W), device=self.device)
        for sample in range(samples):
            noise = torch.empty(img.size()).normal_(0, std).to(self.device)
            noise_image = img + noise
            grad_sum += process(self.get_mask(noise_image, target_class, baseline, steps, process_ig))
        return grad_sum / samples