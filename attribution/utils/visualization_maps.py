import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

@torch.no_grad()
def normalize_saliency(saliency_map, return_device=torch.device('cpu')):
    B, C, H, W = saliency_map.size()
    if C > 1: # the input image is multi-channel
        saliency_map = saliency_map.max(dim=1, keepdim=True)[0]
    saliency_map = F.relu(saliency_map, [0])
    # the shape is B x 1 x H x W, normalize the saliency map along the channel dimension
    saliency_map = saliency_map.view(saliency_map.size(0), -1)
    saliency_map -= saliency_map.min(dim=1, keepdim=True)[0]
    saliency_map /= saliency_map.max(dim=1, keepdim=True)[0]
    saliency_map = saliency_map.view(B, 1, H, W)
    return saliency_map.to(return_device)

def visualize_single_saliency(saliency_map, img_size=None):
    B, C, H, W = saliency_map.size()
    assert B == 1, 'The input saliency map should not be batch inputs'
    if saliency_map.max() > 1 or C > 1:
        saliency_map = normalize_saliency(saliency_map)
    saliency_map = saliency_map.view(H, W, 1)
    saliency_map = saliency_map.cpu().numpy()
    if img_size is not None:
        saliency_map = cv2.resize(saliency_map, (img_size[1], img_size[0]), interpolation=cv2.INTER_LINEAR)
    else:
        saliency_map = cv2.resize(saliency_map, (W, H), interpolation=cv2.INTER_LINEAR)
    saliency_map = cv2.applyColorMap(np.uint8(saliency_map * 255.0), cv2.COLORMAP_JET)
    saliency_map = cv2.cvtColor(saliency_map, cv2.COLOR_BGR2RGB)
    plt.axis('off')
    plt.imshow(saliency_map)
    return saliency_map