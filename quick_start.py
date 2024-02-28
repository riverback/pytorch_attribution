from matplotlib import pyplot as plt
from PIL import Image
import requests
import timm
from timm.data import resolve_model_data_config
from timm.data.transforms_factory import create_transform
import torch

from attribution import BlurIG, GradCAM, CombinedWrapper
from attribution.utils import normalize_saliency, visualize_single_saliency

# Load imagenet labels
IMAGENET_1k_URL = 'https://storage.googleapis.com/bit_models/ilsvrc2012_wordnet_lemmas.txt'
IMAGENET_1k_LABELS = requests.get(IMAGENET_1k_URL).text.strip().split('\n')

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = timm.create_model('resnet50', pretrained=True)
model = model.to(device)
model.eval()
config = resolve_model_data_config(model, None)
transform = create_transform(**config)

# Load image
dog = Image.open('examples/dog.png').convert('RGB')
dog_tensor = transform(dog).unsqueeze(0)
H, W = dog_tensor.shape[-2:]
img = transform(dog).unsqueeze(0)

# We support batch input
img = torch.cat([img, img])
img = img.to(device)
output = model(img)
target_index = torch.argmax(output, dim=1).cpu()
print('Predicted:', IMAGENET_1k_LABELS[target_index[0].item()])

# Gradients visualization
blur_ig_kwargs = {'steps': 100, 
                  'batch_size': 4, 
                  'max_sigma': 50, 
                  'grad_step': 0.01, 
                  'sqrt': False}
blur_ig_net = BlurIG(model)
blur_ig = normalize_saliency(blur_ig_net.get_mask(img, target_index, **blur_ig_kwargs))

# CAM visualization
gradcam_net = GradCAM(model)
gradcam = normalize_saliency(
    gradcam_net.get_mask(img, target_index, target_layer='layer3'))

# Combine Gradients and CAM visualization
combined = CombinedWrapper(model, BlurIG, GradCAM)
combined_saliency = normalize_saliency(
    combined.get_mask(img, target_index, target_layer='layer3', **blur_ig_kwargs))

# Visualize
plt.figure(figsize=(16, 5))
plt.subplot(1, 4, 1)
plt.imshow(dog)
plt.title('Input Image')
plt.axis('off')
plt.subplot(1, 4, 2)
visualize_single_saliency(blur_ig[0].unsqueeze(0))
plt.title('Blur IG')
plt.subplot(1, 4, 3)
visualize_single_saliency(gradcam[0].unsqueeze(0))
plt.title('GradCAM')
plt.subplot(1, 4, 4)
visualize_single_saliency(combined_saliency[0].unsqueeze(0))
plt.title('Combined')
plt.tight_layout()
plt.savefig('examples/quick_start.png', bbox_inches='tight', pad_inches=0.5)