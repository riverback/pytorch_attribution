# Attribution Methods for Timm (PyTorch Image Models)
This is ongoing work to implement various attribution methods for image classification models using [timm](https://github.com/huggingface/pytorch-image-models).

## Gradients Visualization
model: 'convnext_base.fb_in1k'
<img src="./examples/gradients_visualization.png">

## TODO:
- [ ] Unify gradient visualization API
- [ ] Implement CAM visualization for CNN models based on known target_layer names
- [ ] Implement CAM for ViT ,Swin Transformer and etc.
- [ ] Implement CAM visualization for adaptive model architectures (TBD).

## Acknowledgements
This project is inspired by [timm](https://github.com/huggingface/pytorch-image-models), [jacobgil/pytorch-grad-cam](https://github.com/jacobgil/pytorch-grad-cam), [PAIR-code/saliency](https://github.com/PAIR-code/saliency) and [hummat/saliency](https://github.com/hummat/saliency). Thanks for their wonderful work.

