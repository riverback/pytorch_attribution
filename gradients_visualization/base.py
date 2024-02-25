import torch
import numpy as np

class SaliencyMask(object):
    def __init__(self, model):
        self.model = model
        self.model.eval()
        self.gradient = None
        self.hooks = list()
        self.device = next(model.parameters()).device

    def get_mask(self, img, target_class=None):
        raise NotImplementedError(
            'A derived class should implemented this method')

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
            
class VanillaGradient(SaliencyMask):
    def __init__(self, model):
        super(VanillaGradient, self).__init__(model)

    def _encode_one_hot(self, targets, logits):

        one_hot = torch.zeros_like(logits)
        for i in range(0, one_hot.shape[0]):
            one_hot[i, targets[i]] = 1.0
        return one_hot
    
    # return gradients
    def get_mask(self, img: torch.Tensor, target_class: torch.Tensor):
        self.model.zero_grad()
        img = img.clone()
        img.requires_grad = True
        img.retain_grad()

        logits = self.model(img)

        target = self._encode_one_hot(target_class, logits)
        self.model.zero_grad()
        logits.backward(gradient=target, retain_graph=True)
        return img.grad.detach()

    def get_smoothed_mask(self, img, target_class, samples=25, std=0.15, process=lambda x: x**2):
        std = std * (torch.max(img) - torch.min(img)).detach().cpu().numpy()

        B, C, H, W = img.size()
        grad_sum = torch.zeros((B, C, H, W), device=self.device)
        for sample in range(samples):
            noise = torch.empty(img.size()).normal_(0, std).to(self.device)
            noise_image = img + noise
            grad_sum += process(self.get_mask(noise_image, target_class))
        return grad_sum / samples
    
    

    

class BlurIntegratedGradients(VanillaGradient):
    def get_mask(self, img: torch.Tensor, target_class: torch.Tensor, steps=100, max_sigma=50, grad_step=0.01, sqrt=False, batch_size=4):
        ...