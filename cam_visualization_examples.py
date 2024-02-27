from PIL import Image
import matplotlib.pyplot as plt
import torch
import timm
from timm.data import resolve_model_data_config
from timm.data.transforms_factory import create_transform
import requests

from utils import normalize_saliency, visualize_single_saliency
from attribution import GradCAM, GradCAMPlusPlus, XGradCAM, BagCAM


if __name__ == '__main__':
    # Load imagenet labels
    IMAGENET_1k_URL = 'https://storage.googleapis.com/bit_models/ilsvrc2012_wordnet_lemmas.txt'
    IMAGENET_1k_LABELS = requests.get(IMAGENET_1k_URL).text.strip().split('\n')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load a pretrained model
    model = timm.create_model('resnet50', pretrained=True)
    model = model.to(device)
    model.eval()
    config = resolve_model_data_config(model, None)
    transform = create_transform(**config)
    
    # Load an image
    dog = Image.open('examples/dog.png').convert('RGB')
    dog_tensor = transform(dog).unsqueeze(0)
    H, W = dog_tensor.shape[-2:]
    
    # Predict the image
    img = transform(dog).unsqueeze(0)
    img = img.to(device)
    output = model(img)
    target_index = torch.argmax(output[0]).cpu()
    print('Predicted:', IMAGENET_1k_LABELS[target_index.item()])
    
    # Get the target layer
    target_layer_candidates = list()
    for name, module in model.named_modules():
        print(name)
        target_layer_candidates.append(name)
    '''e.g., 
    --ResNet:
    conv1
    bn1
    act1
    maxpool
    layer1
    layer2
    layer3
    layer4
    global_pool
    fc
    
    --Xception41:
    stem
    blocks
    act
    head
    '''
    target_layer = input('Enter the target layer: ')
    while target_layer not in target_layer_candidates:
        print('Invalid layer name')
        target_layer = input('Enter the target layer: ')
    
    # GradCAM
    gradcam_net = GradCAM(model)
    grad_cam = normalize_saliency(gradcam_net.get_mask(img, target_index, target_layer))
    
    # GradCAM++
    gradcam_plus_plus_net = GradCAMPlusPlus(model)
    grad_cam_plus_plus = normalize_saliency(gradcam_plus_plus_net.get_mask(img, target_index, target_layer))

    # XGradCAM
    xgradcam_net = XGradCAM(model)
    xgrad_cam = normalize_saliency(xgradcam_net.get_mask(img, target_index, target_layer))

    # BagCAM
    bagcam_net = BagCAM(model)
    bag_cam = normalize_saliency(bagcam_net.get_mask(img, target_index, target_layer))
    
    