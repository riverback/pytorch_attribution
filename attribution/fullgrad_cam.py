import numpy as np
import torch
from torch.nn import functional as F
from typing import Optional

from .base import CAMWrapper
from .utils import find_layer_predicate_recursive

class FullGrad(CAMWrapper):
    def __init__(self, model: torch.nn.Module, reshape_transform=None):
        if reshape_transform is not None:
            print('Warning: FullGrad may not work properly with ViT and Swin Transformer models.')
        super().__init__(model, reshape_transform)

        self.target_layers = find_layer_predicate_recursive(self.model, self.layer_with_2D_bias)
        self.bias_data = [self.get_bias_data(layer) for layer in self.target_layers]

    def layer_with_2D_bias(self, layer):
        bias_target_layers = [torch.nn.Conv2d, torch.nn.BatchNorm2d]
        if type(layer) in bias_target_layers and layer.bias is not None:
            return True
        return False
    
    def get_bias_data(self, layer):
        if isinstance(layer, torch.nn.BatchNorm2d):
            bias = - (layer.running_mean * layer.weight
                      / torch.sqrt(layer.running_var + layer.eps)) + layer.bias
            return bias.data
        else:
            return layer.bias.data
    
    def get_mask(self, img: torch.Tensor,
                 target_class: torch.Tensor,
                 target_layer: Optional[str] = None):
        if target_layer is not None:
            print('Warning: target_layer is ignored in FullGrad. All bias layers will be used instead')

        B, C, H, W = img.size()
        img = torch.autograd.Variable(img, requires_grad=True)
        self.model.eval()
        self.model.zero_grad()

        # class-specific backpropagation
        logits = self.model(img)
        target = self._encode_one_hot(target_class, logits)
        self.model.zero_grad()
        logits.backward(gradient=target, retain_graph=True)

        target_layer_names = []
        for name, layer in self.model.named_modules():
            if layer in self.target_layers:
                target_layer_names.append(name)
        gradients_list = [self._find(self.gradients, layer_name) for layer_name in target_layer_names]

        input_gradients = img.grad.detach()
        cam_per_target_layer = []

        with torch.no_grad():
            gradient_multiplied_input = input_gradients * img
            gradient_multiplied_input = torch.abs(gradient_multiplied_input)
            gradient_multiplied_input = self.scale_input_across_batch_and_channels(gradient_multiplied_input)
        
            cam_per_target_layer.append(gradient_multiplied_input)
            assert len(gradients_list) == len(self.bias_data)
            for bias, gradients in zip(self.bias_data, gradients_list):
                bias = bias[None, :, None, None]
                bias_grad = torch.abs(bias * gradients)
                bias_grad = self.scale_input_across_batch_and_channels(bias_grad, (H, W))
                bias_grad = bias_grad.sum(dim=1, keepdim=True)
                cam_per_target_layer.append(bias_grad)
            cam_per_target_layer = torch.cat(cam_per_target_layer, dim=1)
            cam = cam_per_target_layer.sum(dim=1, keepdim=True)
            cam = F.relu(cam)
            cam = F.interpolate(cam, (H, W), mode='bilinear', align_corners=False)
            cam = self.normalize_cam(cam)
        
        return cam

    @torch.no_grad()
    def scale_input_across_batch_and_channels(self, input_tensor: torch.Tensor, target_size=None):
        # target_size should be like (H, W)
        B, C, H, W = input_tensor.size()
        input_tensor = input_tensor.view(B, C, -1)
        input_tensor -= input_tensor.min(dim=2, keepdim=True)[0]
        input_tensor /= (input_tensor.max(dim=2, keepdim=True)[0] + 1e-7)
        input_tensor = input_tensor.view(B, C, H, W)
        if target_size is not None:
            input_tensor = F.interpolate(input_tensor, target_size, mode='bilinear', align_corners=False)
        return input_tensor
