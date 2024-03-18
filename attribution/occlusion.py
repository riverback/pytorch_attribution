from .base import Core

import torch


class Occlusion(Core):
    
    @torch.no_grad()
    def get_mask(self, img, target_class, size=15, value=0.0):
        '''
        size: height and width of the occlusion window. 
        value: value to replace values inside the occlusion window with.
        '''
        B, C, H, W = img.size()
        occlusion_scores = torch.zeros_like(img, device=self.device)
        occlusion_window = torch.fill_(torch.zeros((B, C, size, size), device=self.device), value)
        
        original_output = self.model(img)
        for row in range(1 + H - size):
            for col in range(1 + W - size):
                img_occluded = img.clone()
                img_occluded[:, :, row:row+size, col:col+size] = occlusion_window
                output = self.model(img_occluded)
                score_diff = original_output - output
                # the score_diff for the target class
                score_diff = score_diff[torch.arange(B), target_class]
                occlusion_scores[:, :, row:row+size, col:col+size] += score_diff[:, None, None, None]
                
        return occlusion_scores
