from .grad_cam import GradCAM

import torch
from torch.nn import functional as F


class XGradCAM(GradCAM):
    def get_mask(self, img: torch.Tensor,
                 target_class: torch.Tensor,
                 target_layer: str):
        
        B, C, H, W = img.size()
        self.model.eval()
        self.model.zero_grad()
        
        # class-specific backpropagation
        logits = self.model(img)
        target = self._encode_one_hot(target_class, logits)
        self.model.zero_grad()
        logits.backward(gradient=target, retain_graph=True)
        
        # get feature maps and gradients
        feature_maps = self._find(self.feature_maps, target_layer)
        gradients = self._find(self.gradients, target_layer)
        
        # generate CAM
        with torch.no_grad():
            sum_feature_maps = torch.sum(feature_maps, dim=(2, 3))
            sum_feature_maps = sum_feature_maps[:, :, None, None]
            eps = 1e-7
            weights = gradients * feature_maps / (sum_feature_maps + eps)
            weights = torch.sum(weights, dim=(2, 3))
            weights = weights[:, :, None, None]
            cam = torch.mul(feature_maps, weights).sum(dim=1, keepdim=True)
            cam = F.relu(cam)
            cam = F.interpolate(cam, (H, W), mode='bilinear', align_corners=False)
            cam = self.normalize_cam(cam)
            
        return cam