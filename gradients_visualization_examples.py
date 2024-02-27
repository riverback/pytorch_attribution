from PIL import Image
import matplotlib.pyplot as plt
import torch
import timm
from timm.data import resolve_model_data_config
from timm.data.transforms_factory import create_transform
import requests


from attribution import VanillaGradient, IntegratedGradients, BlurIG, GuidedIG
from utils import normalize_saliency, visualize_single_saliency


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
    
    # Support for batch processing
    gradient_net = VanillaGradient(model)
    # shape (2, 3, H, W) before and (2, 1, H, W) after normalize_saliency
    attribution_gradients = normalize_saliency(gradient_net.get_mask(torch.cat([img, img], 0), [target_index]*2))
    
    # Vanilla Gradient
    gradient_net = VanillaGradient(model)
    attribution_gradients = normalize_saliency(gradient_net.get_mask(img, target_index))
    attribution_smooth_gradients = normalize_saliency(gradient_net.get_smoothed_mask(img, target_index, samples=10, std=0.1))
    # Integrated Gradients
    ig_net = IntegratedGradients(model)
    attribution_ig = normalize_saliency(ig_net.get_mask(img, target_index, steps=100))
    attribution_smooth_ig = normalize_saliency(ig_net.get_smoothed_mask(img, target_index, steps=100, std=0.15, samples=10))
    # Blur Integrated Gradients
    blur_ig_net = BlurIG(model)
    attribution_blur_ig = normalize_saliency(blur_ig_net.get_mask(img, target_index, steps=100))
    attribution_smooth_blur_ig = normalize_saliency(blur_ig_net.get_smoothed_mask(img, target_index, steps=100, std=0.15, samples=10))
    # Guided Integrated Gradients
    guided_ig_net = GuidedIG(model)
    attribution_guided_ig = normalize_saliency(guided_ig_net.get_mask(img, target_index, steps=100))
    attribution_smooth_guided_ig = normalize_saliency(guided_ig_net.get_smoothed_mask(img, target_index, steps=100, std=0.15, samples=10))
    
    # Visualize the results
    plt.figure(figsize=(16, 10))
    plt.subplot(2, 5, 1)
    plt.title('Input')
    plt.axis('off')
    plt.imshow(dog)
    plt.subplot(2, 5, 6)
    plt.title('Normalized Input')
    plt.axis('off')
    plt.imshow(img[0].permute(1, 2, 0).cpu().numpy())
    plt.subplot(2, 5, 2)
    plt.title('Vanilla Gradient')
    visualize_single_saliency(attribution_gradients)
    plt.subplot(2, 5, 7)
    plt.title('Smoothed Vanilla Gradient')
    visualize_single_saliency(attribution_smooth_gradients)
    plt.subplot(2, 5, 3)
    plt.title('Integrated Gradients')
    visualize_single_saliency(attribution_ig)
    plt.subplot(2, 5, 8)
    plt.title('Smoothed Integrated Gradients')
    visualize_single_saliency(attribution_smooth_ig)
    plt.subplot(2, 5, 4)
    plt.title('Blur Integrated Gradients')
    visualize_single_saliency(attribution_blur_ig)
    plt.subplot(2, 5, 9)
    plt.title('Smoothed Blur Integrated Gradients')
    visualize_single_saliency(attribution_smooth_blur_ig)
    plt.subplot(2, 5, 5)
    plt.title('Guided Integrated Gradients')
    visualize_single_saliency(attribution_guided_ig)
    plt.subplot(2, 5, 10)
    plt.title('Smoothed Guided Integrated Gradients')
    visualize_single_saliency(attribution_smooth_guided_ig)
    plt.tight_layout()
    plt.savefig('examples/gradients_visualization.png', bbox_inches='tight', pad_inches=0.5)