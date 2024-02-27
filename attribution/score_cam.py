from .base import CAMWrapper

import torch
from torch.nn import functional as F


class ScoreCAM(CAMWrapper):
    def get_mask(self, img: torch.Tensor,
                 target_class: torch.Tensor,
                 target_layer: str,
                 batch_size: int = 16,):
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
        
        with torch.no_grad():
            upsample = torch.nn.UpsamplingBilinear2d(size=(H, W))
            upsampled = upsample(feature_maps)
            maxs = upsampled.view(upsampled.size(0), upsampled.size(1), -1).max(dim=-1)[0]
            mins = upsampled.view(upsampled.size(0), upsampled.size(1), -1).min(dim=-1)[0]
            
            maxs, mins = maxs[:, :, None, None], mins[:, :, None, None]
            upsampled = (upsampled - mins) / (maxs - mins + 1e-8)
            
            imgs = img[:, None, :, :] * upsampled[:, :, None, :, :]
        
        get_targets = lambda o, target: o[target]
        target_class = target_class.cpu().tolist()
        if not isinstance(target_class, list):
            target_class = [target_class]
        
        scores = []
        with torch.no_grad():
            for i in range(imgs.size(0)):
                input_img = imgs[i]
                for batch_i in range(0, input_img.size(0), batch_size):
                    batch = input_img[batch_i: batch_i + batch_size, :]
                    outputs = [get_targets(o, target_class[i]).detach() for o in self.model(batch)]
                    scores.extend(outputs)
            scores = torch.tensor(scores)
            scores = scores.view(feature_maps.shape[0], feature_maps.shape[1])
            weights = F.softmax(scores, dim=-1)
            weights = weights.to(self.device)
            cam = torch.mul(feature_maps, weights[:, :, None, None]).sum(dim=1, keepdim=True)
            cam = F.relu(cam)
            cam = F.interpolate(cam, (H, W), mode='bilinear', align_corners=False)
            cam = self.normalize_cam(cam)
            
        return cam
        