import cv2
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

import torch
from torchvision import models
import torchvision.transforms as T
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.autograd import Function
from torchvision import utils
from torchvision.models.feature_extraction import get_graph_node_names, create_feature_extractor

import timm
from timm.data import resolve_model_data_config
from timm.data.transforms_factory import create_transform

import requests

from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

IMAGENET_1k_URL = 'https://storage.googleapis.com/bit_models/ilsvrc2012_wordnet_lemmas.txt'
IMAGENET_1k_LABELS = requests.get(IMAGENET_1k_URL).text.strip().split('\n')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = timm.create_model('resnet18', pretrained=True)
model = model.to(device)
model.eval()
config = resolve_model_data_config(model, None)
transform = create_transform(**config)

dog = Image.open('examples/dog.png').convert('RGB')
dog_tensor = transform(dog).unsqueeze(0)
H, W = dog_tensor.shape[-2:]

img = transform(dog).unsqueeze(0)
img = img.to(device)
output = model(torch.cat([img, img], 0))
target_index = torch.argmax(output[0]).cpu()
print('Predicted:', IMAGENET_1k_LABELS[target_index.item()])

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

from gradients_visualization import VanillaGradient, IntegratedGradients, BlurIG

from gradients_visualization import BlurIG
blur_ig_net = BlurIG(model)
new_blur_integrated_gradients = blur_ig_net.get_mask(img, torch.tensor([target_index], dtype=torch.long),
                                                 batch_size=4, steps=20)
new_blur_integrated_gradients = normalize_saliency(new_blur_integrated_gradients)
print('blur ig shape:', new_blur_integrated_gradients.shape, new_blur_integrated_gradients.max(), new_blur_integrated_gradients.min())

target_layers = [model.layer4[-1]]
input_tensor = transform(dog).unsqueeze(0).to(device)
cam = XGradCAM(model=model, target_layers=target_layers)
targets = [ClassifierOutputTarget(target_index)]

cam_img = cam(input_tensor=input_tensor, targets=targets)

print('cam_img.shape:', cam_img.shape)