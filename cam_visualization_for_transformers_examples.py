from PIL import Image
import matplotlib.pyplot as plt
import torch
import timm
from timm.data import resolve_model_data_config
from timm.data.transforms_factory import create_transform
import requests

from attribution import GradCAM, GradCAMPlusPlus, XGradCAM, BagCAM, ScoreCAM, LayerCAM, AblationCAM, FullGrad, EigenCAM, EigenGradCAM, HiResCAM
from attribution.utils import normalize_saliency, visualize_single_saliency, get_reshape_transform


if __name__ == '__main__':

    # Load imagenet labels
    IMAGENET_1k_URL = 'https://storage.googleapis.com/bit_models/ilsvrc2012_wordnet_lemmas.txt'
    IMAGENET_1k_LABELS = requests.get(IMAGENET_1k_URL).text.strip().split('\n')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load a pretrained model
    # model = timm.create_model('vit_tiny_patch16_224.augreg_in21k_ft_in1k', pretrained=True)
    model = timm.create_model('swin_base_patch4_window7_224.ms_in22k_ft_in1k', pretrained=True)
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
    target_layer = input('Enter the target layer for swin_base_patch4_window7_224: ')
    while target_layer not in target_layer_candidates:
        print('Invalid layer name')
        target_layer = input('Enter the target layer: ')
    
    # GradCAM
    gradcam_net = GradCAM(model, get_reshape_transform(has_cls_token=False))
    grad_cam = normalize_saliency(gradcam_net.get_mask(img, target_index, target_layer))
    print('GradCAM', grad_cam.shape)
    
    # GradCAM++
    gradcam_plus_plus_net = GradCAMPlusPlus(model, get_reshape_transform(has_cls_token=False))
    grad_cam_plus_plus = normalize_saliency(gradcam_plus_plus_net.get_mask(img, target_index, target_layer))
    print('GradCAM++:', grad_cam_plus_plus.shape)
    
    # HiResCAM
    hirescam_net = HiResCAM(model, get_reshape_transform(has_cls_token=False))
    hires_cam = normalize_saliency(hirescam_net.get_mask(img, target_index, target_layer))
    print('HiResCAM:', hires_cam.shape)

    # XGradCAM
    xgradcam_net = XGradCAM(model, get_reshape_transform(has_cls_token=False))
    xgrad_cam = normalize_saliency(xgradcam_net.get_mask(img, target_index, target_layer))
    print('XGradCAM:', xgrad_cam.shape)
    
    # LayerCAM
    layercam_net = LayerCAM(model, get_reshape_transform(has_cls_token=False))
    layer_cam = normalize_saliency(layercam_net.get_mask(img, target_index, target_layer))
    print('LayerCAM', layer_cam.shape)
    
    # Visualize the saliency maps
    plt.figure(figsize=(18, 10))
    plt.subplot(2,6,1)
    plt.title('Input')
    plt.axis('off')
    plt.imshow(dog)
    plt.subplot(2,6,2)
    plt.title('Swin GradCAM')
    visualize_single_saliency(grad_cam[0].unsqueeze(0))
    plt.subplot(2,6,3)
    plt.title('Swin GradCAM++')
    visualize_single_saliency(grad_cam_plus_plus[0].unsqueeze(0))
    plt.subplot(2,6,4)
    plt.title('Swin HiResCAM')
    visualize_single_saliency(hires_cam[0].unsqueeze(0))
    plt.subplot(2,6,5)
    plt.title('Swin XGradCAM')
    visualize_single_saliency(xgrad_cam[0].unsqueeze(0))
    plt.subplot(2,6,6)
    plt.title('Swin LayerCAM')
    visualize_single_saliency(layer_cam[0].unsqueeze(0))
    
    model = timm.create_model('vit_tiny_patch16_224.augreg_in21k_ft_in1k', pretrained=True)
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
    target_layer = input('Enter the target layer for swin_base_patch4_window7_224: ')
    while target_layer not in target_layer_candidates:
        print('Invalid layer name')
        target_layer = input('Enter the target layer: ')
    
    # GradCAM
    gradcam_net = GradCAM(model, get_reshape_transform(has_cls_token=True))
    grad_cam = normalize_saliency(gradcam_net.get_mask(img, target_index, target_layer))
    print('GradCAM', grad_cam.shape)
    
    # GradCAM++
    gradcam_plus_plus_net = GradCAMPlusPlus(model, get_reshape_transform(has_cls_token=True))
    grad_cam_plus_plus = normalize_saliency(gradcam_plus_plus_net.get_mask(img, target_index, target_layer))
    print('GradCAM++:', grad_cam_plus_plus.shape)
    
    # HiResCAM
    hirescam_net = HiResCAM(model, get_reshape_transform(has_cls_token=True))
    hires_cam = normalize_saliency(hirescam_net.get_mask(img, target_index, target_layer))
    print('HiResCAM:', hires_cam.shape)

    # XGradCAM
    xgradcam_net = XGradCAM(model, get_reshape_transform(has_cls_token=True))
    xgrad_cam = normalize_saliency(xgradcam_net.get_mask(img, target_index, target_layer))
    print('XGradCAM:', xgrad_cam.shape)
    
    # LayerCAM
    layercam_net = LayerCAM(model, get_reshape_transform(has_cls_token=True))
    layer_cam = normalize_saliency(layercam_net.get_mask(img, target_index, target_layer))
    print('LayerCAM', layer_cam.shape)

    plt.subplot(2,6,8)
    plt.title('ViT GradCAM')
    visualize_single_saliency(grad_cam[0].unsqueeze(0))
    plt.subplot(2,6,9)
    plt.title('ViT GradCAM++')
    visualize_single_saliency(grad_cam_plus_plus[0].unsqueeze(0))
    plt.subplot(2,6,10)
    plt.title('ViT HiResCAM')
    visualize_single_saliency(hires_cam[0].unsqueeze(0))
    plt.subplot(2,6,11)
    plt.title('ViT XGradCAM')
    visualize_single_saliency(xgrad_cam[0].unsqueeze(0))
    plt.subplot(2,6,12)
    plt.title('ViT LayerCAM')
    visualize_single_saliency(layer_cam[0].unsqueeze(0))
    
    plt.tight_layout()
    plt.savefig('examples/cam_visualization_for_transformers.png', bbox_inches='tight', pad_inches=0.5)
    
    
    
    