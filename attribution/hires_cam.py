from .base import CAMWrapper

import torch
from torch.nn import functional as F
from typing import List

class HiResCAM(CAMWrapper):
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
            cam = gradients * feature_maps
            cam = cam.sum(dim=1, keepdim=True)
            cam = F.relu(cam)
            cam = F.interpolate(cam, (H, W), mode='bilinear', align_corners=False)
            cam = self.normalize_cam(cam)
            
        return cam