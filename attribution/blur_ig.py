from .base import VanillaGradient

import math
import torch
from torchvision.transforms import GaussianBlur

def gaussian_blur(img: torch.Tensor, sigma: int):
    if sigma == 0:
        return img
    kernel_size = int(4 * sigma + 0.5) + 1
    return GaussianBlur(kernel_size=kernel_size, sigma=sigma)(img)


class BlurIG(VanillaGradient):
    def get_mask(self, img: torch.Tensor, 
                 target_class: torch.Tensor,
                 max_sigma: int = 50,
                 steps: int = 100,
                 grad_step: float = 0.01,
                 sqrt: bool = False,
                 batch_size: int = 4):
        self.model.eval()
        self.model.zero_grad()
        
        if sqrt:
            sigmas = [math.sqrt(float(i) * max_sigma / float(steps)) for i in range(0, steps+1)]
        else:
            sigmas = [float(i) * max_sigma / float(steps) for i in range(0, steps+1)]
        
        step_vector_diff = [sigmas[i+1] - sigmas[i] for i in range(0, steps)]
        total_gradients = torch.zeros_like(img)
        x_step_batched = []
        gaussian_gradient_batched = []

        for i in range(steps):
            with torch.no_grad():
                x_step = gaussian_blur(img, sigmas[i])
                gaussian_gradients = (gaussian_blur(img, sigmas[i] + grad_step) - x_step) / grad_step
            x_step_batched.append(x_step)
            gaussian_gradient_batched.append(gaussian_gradients)
            if len(x_step_batched) == batch_size or i == steps - 1:
                x_step_batched = torch.cat(x_step_batched, dim=0)
                x_step_batched.requires_grad = True
                outputs = torch.softmax(self.model(x_step_batched), dim=1)[:, target_class]
                gradients = torch.autograd.grad(outputs, x_step_batched, torch.ones_like(outputs), create_graph=True)[0]
                gradients = gradients.detach()
                # gradients = super(BlurIG, self).get_mask(x_step_batched, torch.stack([target_class] * x_step_batched.size(0), dim=0))

                with torch.no_grad():
                    total_gradients += (step_vector_diff[i] * 
                                        torch.mul(torch.cat(gaussian_gradient_batched, dim=0), gradients.clone())).sum(dim=0)
                x_step_batched = []
                gaussian_gradient_batched = []
        
        with torch.no_grad():
            blur_ig = total_gradients * -1.0

        return blur_ig
    
    def get_smoothed_mask(self, img: torch.Tensor, 
                          target_class: torch.Tensor, 
                          max_sigma: int = 50,
                          steps: int = 100,
                          grad_step: float = 0.01,
                          sqrt: bool = False,
                          batch_size: int = 4,
                          samples: int = 25,
                          std: float = 0.15,
                          process=lambda x: x**2):
        std = std * (torch.max(img) - torch.min(img)).detach().cpu().numpy()
        
        B, C, H, W = img.size()
        grad_sum = torch.zeros((B, C, H, W), device=self.device)
        for sample in range(samples):
            noise = torch.empty(img.size()).normal_(0, std).to(self.device)
            noise_image = img + noise
            grad_sum += process(self.get_mask(noise_image, target_class, max_sigma, steps, grad_step, sqrt, batch_size))
        return grad_sum / samples
