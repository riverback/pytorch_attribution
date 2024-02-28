import math
import torch

def get_reshape_transform(has_cls_token=False):
    def reshape_transform(x: torch.Tensor):    
        if not has_cls_token:
            # typically is the output of the swin transformer
            x = x.permute(0, 3, 1, 2)
            return x
        # typically is the output of the ViT
        B, token_numbers, patch_size = x.size()
        if has_cls_token:
            token_numbers -= 1
            x = x[:, 1:, :]
        
        img_width = int(math.sqrt(token_numbers))
        assert img_width * img_width == token_numbers
        x = x.view(B, img_width, img_width, patch_size)
        x = x.permute(0, 3, 1, 2)
        return x
    
    return reshape_transform