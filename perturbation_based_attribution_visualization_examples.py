from PIL import Image
import matplotlib.pyplot as plt
import torch
import timm
from timm.data import resolve_model_data_config
from timm.data.transforms_factory import create_transform
import requests


from attribution import Occlusion
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
    
    # Occlusion
    occlusion_net = Occlusion(model)
    occlusion = normalize_saliency(occlusion_net.get_mask(img, target_index))
    occlusion_2 = normalize_saliency(occlusion_net.get_mask(img, target_index, size=30))
    
    # Visualize the results
    plt.figure(figsize=(16, 5))
    plt.subplot(1, 3, 1)
    plt.title('Input')
    plt.axis('off')
    plt.imshow(dog)
    plt.subplot(1, 3, 2)
    plt.title('Occlusion window-15')
    visualize_single_saliency(occlusion[0].unsqueeze(0))
    plt.subplot(1, 3, 3)
    plt.title('Occlusion window-30')
    visualize_single_saliency(occlusion_2[0].unsqueeze(0))
    plt.tight_layout()
    plt.savefig('examples/perturbation_based_visualization.png', bbox_inches='tight', pad_inches=0.5)