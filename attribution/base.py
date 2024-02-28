import torch
from typing import Optional, List, Union
from .utils import normalize_saliency

class Core(torch.nn.Module):
    def __init__(self, model: torch.nn.Module):
        super(Core, self).__init__()
        self.model = model
        self.hooks = list() # list of hooks
        self.device = next(model.parameters()).device
        
    def _encode_one_hot(self, targets: torch.Tensor, logits: torch.Tensor):
        if not isinstance(targets, torch.Tensor):
            targets = torch.tensor([targets], device=self.device)
        targets = targets.view(logits.size(0))
        one_hot = torch.zeros_like(logits)
        for i in range(0, one_hot.shape[0]):
            one_hot[i, targets[i]] = 1.0
        return one_hot

    def get_mask(self, img, target_class=None):
        raise NotImplementedError(
            'A derived class should implemented this method')

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
            
    def forward(self, img):
        logits = self.model(img)
        return logits

class VanillaGradient(Core):
    def __init__(self, model):
        super(VanillaGradient, self).__init__(model)

    # return gradients
    def get_mask(self, img: torch.Tensor, target_class: torch.Tensor) -> torch.Tensor:
        self.model.eval()
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


class CAMWrapper(Core):
    def __init__(self, model: torch.nn.Module, reshape_transform=None):
        super(CAMWrapper, self).__init__(model)
    
        self.feature_maps = dict()
        self.gradients = dict()
        self.reshape_transform = reshape_transform
        
        def save_feature_maps(name):
            def forward_hook(module, input, output):
                self.feature_maps[name] = output.detach()

            return forward_hook
        
        def save_gradients(name):
            def _store_grad(grad):
                self.gradients[name] = grad.detach()
            def forward_hook(module, input, output):
                if not hasattr(output, 'requires_grad') or not output.requires_grad:
                    return
                output.register_hook(_store_grad)
                
            return forward_hook
                
        for name, module in self.model.named_modules():
            self.hooks.append(module.register_forward_hook(save_feature_maps(name)))
            self.hooks.append(module.register_forward_hook(save_gradients(name)))
                
    def _find(self, saved_dict, name: str):
        if name in saved_dict.keys():
            if self.reshape_transform is not None:
                return self.reshape_transform(saved_dict[name])
            return saved_dict[name]
        
        raise ValueError('Invalid layer name')
    
    def get_mask(self, img: torch.Tensor,
                 target_class: torch.Tensor,
                 target_layer: Union[str, List[str]]):
        raise NotImplementedError('A derived class should implemented this method')
    
    @torch.no_grad()
    def normalize_cam(self, cam: torch.Tensor):
        B, C, H, W = cam.size()
        cam = cam.view(cam.size(0), -1)
        cam -= cam.min(dim=1, keepdim=True)[0]
        cam = cam / (cam.max(dim=1, keepdim=True)[0]+1e-5)
        return cam.view(B, C, H, W)
    
    
class CombinedWrapper(Core):
    def __init__(self, model: torch.nn.Module, gradient_net: Core, cam_net: CAMWrapper):
        super(CombinedWrapper, self).__init__(model)
        self.gradient_net = gradient_net(model)
        self.cam_net = cam_net(model)
        
    def get_mask(self, img: torch.Tensor, target_class: torch.Tensor, target_layer: str, **kwargs_for_gradient_net):
        B, C, H, W = img.size()
        self.model.eval()
        self.model.zero_grad()
        cam = self.cam_net.get_mask(img, target_class, target_layer)
        gradients = self.gradient_net.get_mask(img, target_class, **kwargs_for_gradient_net)
        gradients = normalize_saliency(gradients, return_device=self.device)
        
        with torch.no_grad():
            attribution = gradients * cam
            attribution = torch.nn.functional.relu(attribution)
            attribution = self.cam_net.normalize_cam(attribution)
            
        return attribution