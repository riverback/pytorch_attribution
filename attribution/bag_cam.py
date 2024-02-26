from .base import CAMWrapper

import torch
from torch.nn import functional as F


class BagCAM(CAMWrapper):
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
            term_2 = gradients * feature_maps
            term_1 = term_2 + 1
            term_1 = F.adaptive_avg_pool2d(term_1, 1)
            cam = F.relu(torch.mul(term_1, term_2)).sum(dim=1, keepdim=True)
            cam = F.interpolate(cam, (H, W), mode='bilinear', align_corners=False)
            cam = self.normalize_cam(cam)
            
        return cam