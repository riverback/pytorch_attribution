from .base import CAMWrapper
from .utils import replace_layer_recursive, get_2d_projection

import numpy as np
import torch
from torch.nn import functional as F

class AblationLayer(torch.nn.Module):
    def __init__(self):
        super(AblationLayer, self).__init__()

    def objectiveness_mask_from_svd(self, feature_maps, threshold=0.01):
        projection = get_2d_projection(feature_maps[None, :])[0, :]
        projection = np.abs(projection)
        projection = projection - projection.min()
        projection = projection / projection.max()
        projection = projection > threshold
        return projection

    def feature_maps_to_be_ablated(
            self,
            feature_maps,
            ratio_channels_to_ablate=1.0):
        if ratio_channels_to_ablate == 1.0:
            self.indices = np.int32(range(feature_maps.shape[0]))
            return self.indices

        projection = self.objectiveness_mask_from_svd(feature_maps)

        scores = []
        for channel in feature_maps:
            normalized = np.abs(channel)
            normalized = normalized - normalized.min()
            normalized = normalized / np.max(normalized)
            score = (projection * normalized).sum() / normalized.sum()
            scores.append(score)
        scores = np.float32(scores)

        indices = list(np.argsort(scores))
        high_score_indices = indices[::-
                                     1][: int(len(indices) *
                                              ratio_channels_to_ablate)]
        low_score_indices = indices[: int(
            len(indices) * ratio_channels_to_ablate)]
        self.indices = np.int32(high_score_indices + low_score_indices)
        return self.indices

    def set_next_batch(
            self,
            input_batch_index,
            feature_maps: torch.Tensor,
            num_channels_to_ablate):
        self.feature_maps_ablation_layer = feature_maps[input_batch_index, :, :, :].clone(
        ).unsqueeze(0).repeat(num_channels_to_ablate, 1, 1, 1)

    def __call__(self, x):
        output = self.feature_maps_ablation_layer
        for i in range(output.size(0)):
            # Commonly the minimum activation will be 0,
            # And then it makes sense to zero it out.
            # However depending on the architecture,
            # If the values can be negative, we use very negative values
            # to perform the ablation, deviating from the paper.
            if torch.min(output) == 0:
                output[i, self.indices[i], :] = 0
            else:
                ABLATION_VALUE = 1e7
                output[i, self.indices[i], :] = torch.min(
                    output) - ABLATION_VALUE

        return output

class AblationCAM(CAMWrapper):
    def get_mask(self, img: torch.Tensor,
                 target_class: torch.Tensor,
                 target_layer: str,
                 batch_size: int = 32,
                 ratio_channels_to_ablate: float = 1.0):
        
        B, C, H, W = img.size()
        self.model.eval()
        self.model.zero_grad()

        # class-specific backpropagation
        logits = self.model(img)
        target = self._encode_one_hot(target_class, logits)
        self.model.zero_grad()
        logits.backward(gradient=target, retain_graph=True)

        get_targets = lambda o, target: o[target]
        target_class = target_class.cpu().tolist()
        if not isinstance(target_class, list):
            target_class = [target_class]

        original_scores = [get_targets(o, t).cpu().detach() for o, t in zip(logits, target_class)]
        feature_maps = self._find(self.feature_maps, target_layer)

        # save original layer and replace the model back to the original state later
        original_target_layer = self.get_target_module(target_layer)
        ablation_layer = AblationLayer()
        replace_layer_recursive(self.model, original_target_layer, ablation_layer)

        # get weights
        number_of_channels = feature_maps.size(1)
        weights = []
        with torch.no_grad():
            for batch_idx, (target, I) in enumerate(zip(target_class, img)):
                new_scores = []
                batch_tensor = I.repeat(batch_size, 1, 1, 1)

                channels_to_ablate = ablation_layer.feature_maps_to_be_ablated(
                    feature_maps[batch_idx, :], ratio_channels_to_ablate
                )
                number_of_channels_to_ablate = len(channels_to_ablate)

                for i in range(0, number_of_channels_to_ablate, batch_size):
                    if i + batch_size > number_of_channels_to_ablate:
                        batch_tensor = batch_tensor[:(number_of_channels_to_ablate - i)]
                    
                    ablation_layer.set_next_batch(
                        batch_idx, feature_maps, batch_tensor.size(0)
                    )

                    new_scores.extend([get_targets(o, target).cpu().detach() for o in self.model(batch_tensor)])
                    ablation_layer.indices = ablation_layer.indices[batch_size:]
                
                new_scores = self.assemble_ablation_scores(
                    new_scores, original_scores[batch_idx], channels_to_ablate, number_of_channels
                )
                weights.extend(new_scores)

            weights = np.float32(weights)
            weights = weights.reshape(feature_maps.shape[:2])
            original_scores = np.array(original_scores)[:, None]
            weights = (original_scores - weights) / original_scores
            weights = torch.from_numpy(weights).to(self.device)[:, :, None, None]

            cam = torch.mul(feature_maps, weights).sum(dim=1, keepdim=True)
            cam = F.relu(cam)
            cam = F.interpolate(cam, (H, W), mode='bilinear', align_corners=False)
            cam = self.normalize_cam(cam)

            replace_layer_recursive(self.model, ablation_layer, original_target_layer)
            
        return cam
    
    def get_target_module(self, target_layer: str):
        for name, module in self.model.named_modules():
            if name == target_layer:
                return module
        raise ValueError(f"Layer {target_layer} not found in model")
    
    def assemble_ablation_scores(self,
                                 new_scores: list,
                                 original_score: float,
                                 ablated_channels: np.ndarray,
                                 number_of_channels: int) -> np.ndarray:
        """ Take the value from the channels that were ablated,
            and just set the original score for the channels that were skipped """

        index = 0
        result = []
        sorted_indices = np.argsort(ablated_channels)
        ablated_channels = ablated_channels[sorted_indices]
        new_scores = np.float32(new_scores)[sorted_indices]

        for i in range(number_of_channels):
            if index < len(ablated_channels) and ablated_channels[index] == i:
                weight = new_scores[index]
                index = index + 1
            else:
                weight = original_score
            result.append(weight)

        return result