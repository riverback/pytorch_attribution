import torch
from torch.nn import functional as F

from .base import CombinedWrapper
from .grad_cam import GradCAM
from .guided_backprop import GuidedBackProp

class GuidedGradCAM(CombinedWrapper):
    def __init__(self, model: torch.nn.Module, reshape_transform=None):
        gradient_net = GuidedBackProp
        cam_net = GradCAM
        super(GuidedGradCAM, self).__init__(model, gradient_net, cam_net, reshape_transform)