from .base import VanillaGradient

import torch
import numpy as np
import math

eps = 1e-9

@torch.no_grad()
def l1_distance(x, y):
    return torch.abs(x - y).sum()

@torch.no_grad()
def translate_x_to_alpha(x: torch.Tensor, x_input: torch.Tensor, x_baseline: torch.Tensor):
    '''Alpha shows the relative position of x in the straight line between x_baseline and x_input'''
    alpha = torch.where(x_input-x_baseline != 0, 
                        (x - x_baseline) / (x_input - x_baseline + eps), torch.tensor(np.nan).to(x.device))
    return alpha

@torch.no_grad()
def translate_alpha_to_x(alpha: torch.Tensor, x_input: torch.Tensor, x_baseline: torch.Tensor):
    '''Translate alpha to x'''
    assert 0.0 <= alpha <= 1.0, 'Alpha should be in the range [0, 1]'
    x = x_baseline + alpha * (x_input - x_baseline)
    return x

class GuidedIG(VanillaGradient):
    def get_mask(self, img: torch.Tensor, 
                 target_class: torch.Tensor,
                 baseline='None',
                 steps=128,
                 fraction=0.25,
                 max_dist=0.02):
        
        self.model.eval()
        self.model.zero_grad()
        
        if baseline is None or baseline == 'None':
            baseline = torch.zeros_like(img, device=self.device)
        elif baseline == 'black':
            baseline = torch.ones_like(img, device=self.device) * torch.min(img).detach()
        elif baseline == 'white':
            baseline = torch.ones_like(img, device=self.device) * torch.max(img).detach()
        else:
            raise ValueError(f'Baseline {baseline} is not supported, use "black", "white" or None')
        
        return self.guided_ig_impl(img, target_class, baseline, steps, fraction, max_dist)
    
    def get_smoothed_mask(self, img: torch.Tensor,
                          target_class: torch.Tensor,
                          baseline = 'None',
                          steps: int = 128,
                          fraction: float = 0.25,
                          max_dist: float = 0.02,
                          samples: int = 25,
                          std: float = 0.15,
                          process=lambda x: x**2):
        std = std * (torch.max(img) - torch.min(img)).detach().cpu().numpy()
        
        B, C, H, W = img.size()
        grad_sum = torch.zeros((B, C, H, W), device=self.device)
        for sample in range(samples):
            noise = torch.empty(img.size()).normal_(0, std).to(self.device)
            noise_image = img + noise
            grad_sum += process(self.get_mask(noise_image, target_class, baseline, steps, fraction, max_dist))
        return grad_sum / samples
    
    def guided_ig_impl(self, img: torch.Tensor,
                       target_class: torch.Tensor,
                       baseline: torch.Tensor,
                       steps: int,
                       fraction: float,
                       max_dist: float):
        guided_ig = torch.zeros_like(img, dtype=torch.float64, device=self.device)

        x = baseline.clone()

        l1_total = l1_distance(baseline, img)
        if l1_total.sum() < eps:
            return guided_ig
        
        for step in range(steps):
            gradients_actual = super().get_mask(x, target_class)
            gradients = gradients_actual.clone().detach()
            alpha = (step + 1.0) / steps
            alpha_min = max(alpha - max_dist, 0.0)
            alpha_max = min(alpha + max_dist, 1.0)
            x_min = translate_alpha_to_x(alpha_min, img, baseline).detach()
            x_max = translate_alpha_to_x(alpha_max, img, baseline).detach()

            with torch.no_grad():
                l1_target = l1_total * (1 - (step + 1) / steps)

            # Iterate until the desired L1 distance has been reached.
            gamma = torch.tensor(np.inf)
            while gamma > 1.0:
                x_old = x.clone().detach()
                x_alpha = translate_x_to_alpha(x, img, baseline).detach()
                x_alpha[torch.isnan(x_alpha)] = alpha_max
                x.requires_grad = False
                x[x_alpha < alpha_min] = x_min[x_alpha < alpha_min]
                
                l1_current = l1_distance(x, img)

                if math.isclose(l1_target, l1_current, rel_tol=eps, abs_tol=eps):
                    with torch.no_grad():
                        guided_ig += gradients_actual * (x - x_old)
                    break

                gradients[x == x_max] = torch.tensor(np.inf)
                
                threshold = torch.quantile(torch.abs(gradients), fraction, interpolation='lower')
                s = torch.logical_and(torch.abs(gradients) <= threshold, gradients != torch.tensor(np.inf))

                with torch.no_grad():
                    l1_s = (torch.abs(x - x_max) * s).sum()

                    if l1_s > 0:
                        gamma = (l1_current - l1_target) / l1_s
                    else:
                        gamma = torch.tensor(np.inf)

                if gamma > 1.0:
                    x[s] = x_max[s]
                else:
                    assert gamma >= 0.0, f'Gamma should be non-negative, but got {gamma.min()}'
                    x[s] = translate_alpha_to_x(gamma, x_max, x)[s]

                with torch.no_grad():
                    guided_ig += gradients_actual * (x - x_old)

        return guided_ig


        
        