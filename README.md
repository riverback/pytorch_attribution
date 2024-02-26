# Explainability Methods for PyTorch Image Models (timm)
This is an ongoing work to implement various attribution methods for image classification models from [timm](https://github.com/huggingface/pytorch-image-models).

## Gradients Visualization
results of resnet50 from timm, example code at [./gradientss_visualization_examples.py](./gradients_visualization_examples.py)

<img src="./examples/gradients_visualization.png">


## TODO:
- [x] Unify gradient visualization API
- [ ] Documentation for gradient visualization
- [ ] Implement CAM visualization for CNN models based on known target_layer names
- [ ] Implement CAM for ViT ,Swin Transformer and etc.
- [ ] Documentation for CAM visualization
- [ ] Unify all APIs
- [ ] Implement CAM visualization for adaptive model architectures (TBD).

## Acknowledgements
This project is inspired by [timm](https://github.com/huggingface/pytorch-image-models), [jacobgil/pytorch-grad-cam](https://github.com/jacobgil/pytorch-grad-cam), [PAIR-code/saliency](https://github.com/PAIR-code/saliency) and [hummat/saliency](https://github.com/hummat/saliency). Thanks for their wonderful work.

