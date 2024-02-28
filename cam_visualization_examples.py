from PIL import Image
import matplotlib.pyplot as plt
import torch
import timm
from timm.data import resolve_model_data_config
from timm.data.transforms_factory import create_transform
import requests

from attribution import GradCAM, GradCAMPlusPlus, XGradCAM, BagCAM, ScoreCAM, LayerCAM, AblationCAM, FullGrad, EigenCAM, EigenGradCAM, HiResCAM
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
    
    # GradCAM
    gradcam_net = GradCAM(model)
    grad_cam = normalize_saliency(gradcam_net.get_mask(img, target_index, target_layer))
    print('GradCAM', grad_cam.shape)
    
    # GradCAM++
    gradcam_plus_plus_net = GradCAMPlusPlus(model)
    grad_cam_plus_plus = normalize_saliency(gradcam_plus_plus_net.get_mask(img, target_index, target_layer))
    print('GradCAM++:', grad_cam_plus_plus.shape)
    
    # HiResCAM
    hirescam_net = HiResCAM(model)
    hires_cam = normalize_saliency(hirescam_net.get_mask(img, target_index, target_layer))
    print('HiResCAM:', hires_cam.shape)

    # XGradCAM
    xgradcam_net = XGradCAM(model)
    xgrad_cam = normalize_saliency(xgradcam_net.get_mask(img, target_index, target_layer))
    print('XGradCAM:', xgrad_cam.shape)

    # BagCAM
    bagcam_net = BagCAM(model)
    bag_cam = normalize_saliency(bagcam_net.get_mask(img, target_index, target_layer))
    print('BagCAM:', bag_cam.shape)
    
    # ScoreCAM
    scorecam_net = ScoreCAM(model)
    score_cam = normalize_saliency(scorecam_net.get_mask(img, target_index, target_layer))
    print('ScoreCAM', score_cam.shape)
    
    # EigenCAM
    eigencam_net = EigenCAM(model)
    eigen_cam = normalize_saliency(eigencam_net.get_mask(img, target_index, target_layer))
    print('EigenCAM', eigen_cam.shape)
    
    # EigenGradCAM
    eigengradcam_net = EigenGradCAM(model)
    eigen_grad_cam = normalize_saliency(eigengradcam_net.get_mask(img, target_index, target_layer))
    print('EigenGradCAM', eigen_grad_cam.shape)
    
    # LayerCAM
    layercam_net = LayerCAM(model)
    layer_cam = normalize_saliency(layercam_net.get_mask(img, target_index, target_layer))
    print('LayerCAM', layer_cam.shape)

    # AblationCAM
    ablationcam_net = AblationCAM(model)
    ablation_cam = normalize_saliency(ablationcam_net.get_mask(img, target_index, target_layer))
    print('AblationCAM', ablation_cam.shape)

    # FullGrad
    fullgrad_net = FullGrad(model)
    full_grad = normalize_saliency(fullgrad_net.get_mask(img, target_index, target_layer=None))
    print('FullGrad', full_grad.shape)
    
    # Visualize the saliency maps
    plt.figure(figsize=(16, 15))
    plt.subplot(3,5,1)
    plt.title('Input')
    plt.axis('off')
    plt.imshow(dog)
    plt.subplot(3,5,2)
    plt.title('GradCAM')
    visualize_single_saliency(grad_cam[0].unsqueeze(0))
    plt.subplot(3,5,3)
    plt.title('GradCAM++')
    visualize_single_saliency(grad_cam_plus_plus[0].unsqueeze(0))
    plt.subplot(3,5,4)
    plt.title('HiResCAM')
    visualize_single_saliency(hires_cam[0].unsqueeze(0))
    plt.subplot(3,5,5)
    plt.title('FullGrad')
    visualize_single_saliency(full_grad[0].unsqueeze(0))
    plt.subplot(3,5,6)
    plt.title('AblationCAM')
    visualize_single_saliency(ablation_cam[0].unsqueeze(0))
    plt.subplot(3,5,7)
    plt.title('ScoreCAM')
    visualize_single_saliency(score_cam[0].unsqueeze(0))
    plt.subplot(3,5,8)
    plt.title('EigenCAM')
    visualize_single_saliency(eigen_cam[0].unsqueeze(0))
    plt.subplot(3,5,9)
    plt.title('EigenGradCAM')
    visualize_single_saliency(eigen_grad_cam[0].unsqueeze(0))
    plt.subplot(3,5,10)
    plt.title('XGradCAM')
    visualize_single_saliency(xgrad_cam[0].unsqueeze(0))
    plt.subplot(3,5,11)
    plt.title('LayerCAM')
    visualize_single_saliency(layer_cam[0].unsqueeze(0))
    plt.subplot(3,5,12)
    plt.title('BagCAM')
    visualize_single_saliency(bag_cam[0].unsqueeze(0))
    
    
    plt.tight_layout()
    plt.savefig('examples/cam_visualization.png', bbox_inches='tight', pad_inches=0.5)
    
    
    
    