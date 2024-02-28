from PIL import Image
import matplotlib.pyplot as plt
import torch
import timm
from timm.data import resolve_model_data_config
from timm.data.transforms_factory import create_transform
import requests

from attribution import GradCAM, GradCAMPlusPlus, XGradCAM, BagCAM, ScoreCAM, LayerCAM, AblationCAM, FullGrad, EigenCAM, EigenGradCAM, HiResCAM
from attribution import VanillaGradient, GuidedBackProp, IntegratedGradients, BlurIG, GuidedIG
from attribution import CombinedWrapper, GuidedGradCAM
from attribution.utils import normalize_saliency, visualize_single_saliency


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
    img = torch.cat([img, img])
    img = img.to(device)
    output = model(img)
    target_index = torch.argmax(output, dim=1).cpu()
    print('Predicted:', IMAGENET_1k_LABELS[target_index[0].item()])
    
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
        
    # Guided Grad-CAM
    net = GuidedGradCAM(model)
    guided_gradcam = normalize_saliency(net.get_mask(img, target_index, target_layer))
    print('Guided Grad-CAM', guided_gradcam.shape)
    
    # GuidedBackprop + FullGrad
    net = CombinedWrapper(model, GuidedBackProp, FullGrad)
    guided_fullgrad = normalize_saliency(net.get_mask(img, target_index, target_layer))
    print('GuidedBackProp + FullGrad', guided_fullgrad.shape)
    
    # BlurIG + EigenCAM
    net = CombinedWrapper(model, BlurIG, EigenCAM)
    kwargs = {'steps': 20}
    blurig_eigencam = normalize_saliency(net.get_mask(img, target_index, target_layer, **kwargs))
    print('BlurIG + EigenCAM', blurig_eigencam.shape)
    
    # Visualize the results
    plt.figure(figsize=(16, 5))
    plt.subplot(1, 4, 1)
    plt.title('Input')
    plt.axis('off')
    plt.imshow(dog)
    plt.subplot(1, 4, 2)
    plt.title('Guided Grad-CAM')
    visualize_single_saliency(guided_gradcam[0].unsqueeze(0))
    plt.subplot(1, 4, 3)
    plt.title('GuidedBackProp + FullGrad')
    visualize_single_saliency(guided_fullgrad[0].unsqueeze(0))
    plt.subplot(1, 4, 4)
    plt.title('BlurIG + EigenCAM')
    visualize_single_saliency(blurig_eigencam[0].unsqueeze(0))
    
    plt.tight_layout()
    plt.savefig('examples/combine_cam_and_gradients_visualization.png', bbox_inches='tight', pad_inches=0.5)