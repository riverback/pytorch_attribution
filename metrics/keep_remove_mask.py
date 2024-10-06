import torch
from torchvision.transforms import GaussianBlur
import numpy as np

import json
import numpy as np
#import shap
#import shap.benchmark as benchmark
import scipy as sp
import math
from collections import OrderedDict
import numbers
import torch
from torch import nn
from torch.nn import functional as F
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import itertools
from tqdm import tqdm



class GaussianSmoothing(nn.Module):
    """
    Apply gaussian smoothing on a
    1d, 2d or 3d tensor. Filtering is performed seperately for each channel
    in the input using a depthwise convolution.
    Arguments:
        channels (int, sequence): Number of channels of the input tensors. Output will
            have this number of channels as well.
        kernel_size (int, sequence): Size of the gaussian kernel.
        sigma (float, sequence): Standard deviation of the gaussian kernel.
        dim (int, optional): The number of dimensions of the data.
            Default value is 2 (spatial).
    """
    def __init__(self, device, channels, kernel_size, sigma, dim=2):
        super(GaussianSmoothing, self).__init__()
        if isinstance(kernel_size, numbers.Number):
            kernel_size = [kernel_size] * dim
        if isinstance(sigma, numbers.Number):
            sigma = [sigma] * dim

        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid(
            [
                torch.arange(size, dtype=torch.float32)
                for size in kernel_size
            ]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= 1 / (std * math.sqrt(2 * math.pi)) * \
                      torch.exp(-((mgrid - mean) / (2 * std)) ** 2)

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer('weight', kernel)
        self.weight = self.weight.to(device)
        self.groups = channels

        if dim == 1:
            self.conv = F.conv1d
        elif dim == 2:
            self.conv = F.conv2d
        elif dim == 3:
            self.conv = F.conv3d
        else:
            raise RuntimeError(
                'Only 1, 2 and 3 dimensions are supported. Received {}.'.format(dim)
            )

    def forward(self, input):
        """
        Apply gaussian filter to input.
        Arguments:
            input (torch.Tensor): Input to apply gaussian filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """
        return self.conv(input, weight=self.weight, groups=self.groups)

@torch.no_grad()
def attribution_to_mask(attribution, percent_unmasked, sort_order, perturbation, device):

    attribution = attribution.clone().detach()
    attribution = attribution.mean(dim=1)
    attribution = attribution.unsqueeze(1)
    
    zeros = torch.zeros(attribution.shape).to(device)
    
    #invert attribution for negative case
    if sort_order == 'negative' or sort_order == 'negative_target' or sort_order == 'negative_topk':
        attribution = -attribution
        
    if sort_order == 'absolute':
        attribution = torch.abs(attribution)
    
    #for positive and negative the negative and positive values are always masked ond not considered for the topk
    positives = torch.maximum(attribution, zeros)
    nb_positives = torch.count_nonzero(positives)
    
    orig_shape = positives.size()
    positives = positives.view(positives.size(0), 1, -1)
    nb_pixels = positives.size(2)
    
    if perturbation == 'keep':
        # find features to keep
        ret = torch.topk(positives, k=int(torch.minimum(torch.tensor(percent_unmasked*nb_pixels).to(device), nb_positives)), dim=2)
        
    if perturbation == 'remove':
        #set zeros to large value
        positives_wo_zero = positives.clone()
        positives_wo_zero[positives_wo_zero == 0.] = float("Inf")
        # find features to keep
        ret = torch.topk(positives_wo_zero, k=int(torch.minimum(torch.tensor(percent_unmasked*nb_pixels).to(device), nb_positives)), dim=2, largest=False)
    ret.indices.shape   
    # Scatter to zero'd tensor
    res = torch.zeros_like(positives)
    res.scatter_(2, ret.indices, ret.values)
    res = res.view(*orig_shape)

    res = (res == 0).float() # set topk values to zero and all zeros to one
    res = res.repeat(1,3,1,1)
    return res

@torch.no_grad()
def keep_remove_metrics(sort_order, perturbation, model, images, labels, attribution, device, step_num=16):
    options = [(sort_order, perturbation)]
    vals = {}

    smoothing = GaussianSmoothing(device, 3, 51, 41)
    percent_unmasked_range = np.geomspace(0.01, 1.0, num=step_num)
    
    for percent_unmasked in percent_unmasked_range:
        vals[perturbation + "_" + sort_order + "_" + str(percent_unmasked)] = 0.0

    masks = []
    for sort_order, perturbation in options:
        for percent_unmasked in percent_unmasked_range:
            #create masked images
            for sample in range(attribution.shape[0]):
                mask = attribution_to_mask(attribution[sample].unsqueeze(0), percent_unmasked, sort_order, perturbation, device)
                masks.append(mask)

    mask = torch.cat(masks, dim=0)

    images_masked_pt = images.clone().repeat(int(mask.shape[0]/images.size(0)), 1, 1, 1)
    images_smoothed_pt = images.clone().repeat(int(mask.shape[0]/images.size(0)), 1, 1, 1)
    images_smoothed_pt = F.pad(images_smoothed_pt, (25,25,25,25), mode='reflect')
    images_smoothed_pt = smoothing(images_smoothed_pt)
    images_masked_pt[mask.bool()] = images_smoothed_pt[mask.bool()]

    #images_masked = normalize(torch.tensor(images_masked_np / 255.).unsqueeze(0).permute(0,3,1,2))
    images_masked = images_masked_pt
    images_masked = images_masked.to(device)
    out_masked = model(images_masked)
    out_masked = out_masked.softmax(dim=-1)

    #split out_masked in the chunks that correspond to the individual run
    option_runs = torch.split(out_masked, int(out_masked.shape[0]/len(options)))
    for o, (sort_order, perturbation) in enumerate(options):
        option_run = option_runs[o] # N, 1000
        percent_unmasked_runs = torch.split(option_run, int(option_run.shape[0]/len(percent_unmasked_range)))  # N, 1000
        for p, percent_unmasked in enumerate(percent_unmasked_range):
            percent_unmasked_run = percent_unmasked_runs[p] # N, 1000
            #if len(percent_unmasked_run.shape) == 1:
            #    percent_unmasked_run = percent_unmasked_run.unsqueeze(0)

            if sort_order == 'positive':
                vals[perturbation + "_" + sort_order + "_" + str(percent_unmasked)] += torch.gather(percent_unmasked_run, 1, labels.unsqueeze(-1)).sum().cpu().item()
            if sort_order == 'negative':
                vals[perturbation + "_" + sort_order + "_" + str(percent_unmasked)] += torch.gather(percent_unmasked_run, 1, labels.unsqueeze(-1)).sum().cpu().item()
            if sort_order == 'absolute':
                correct = (torch.max(percent_unmasked_run, 1)[1] == labels).float().sum().cpu().item()
                vals[perturbation + "_" + sort_order + "_" + str(percent_unmasked)] += correct

    for sort_order, perturbation in options:   
        for percent_unmasked in percent_unmasked_range:
            vals[perturbation + "_" + sort_order + "_" + str(percent_unmasked)] /= images.shape[0]

    for sort_order, perturbation in options:
        xs = []
        ys = []
        for percent_unmasked in percent_unmasked_range:
            xs.append(percent_unmasked)
            ys.append(vals[perturbation + "_" + sort_order + "_" + str(percent_unmasked)])
        auc = np.trapz(ys, xs)
        xs = np.array(xs)
        ys = np.array(ys)
    return auc, vals, xs, ys


def kpm(original_input, attribution, target_label, forward, step=0.05):
    """ kpm: keep positive mask, 
        increase the numbers of input pixels that attribution is positive from high attribution to low attribution, other pixels are masked using the average value of masked pixels
        record acc score of the target label of each step, finally calculate the area under the curve
        the output acc is expected to increase from 0 to the acc of the original input
    original_input: batch original input, B*C*H*W
    attribution: batch attribution, B*1*H*W
    mask: batch target mask, B*1*H*W
    target_label: batch target label, B
    forward: forward function, forward(masked_input, target_label), return mean target score of the batch and accuracy of the batch
    step: step of masking attributed pixels, percentage of the input features of original input
    """
    B, C, H, W = original_input.size()
    # get sampling interval
    max_attribution = attribution.max()
    min_attribution = 0. # attribution.min()
    interval = (max_attribution - min_attribution) * step
    N = int(1 / step) + 1
    
    y_axis = []
    
    # get blurred input
    # blurred_original_input = GaussianBlur(51, 41)(original_input)
    blurred_original_input = torch.zeros_like(original_input, device=original_input.device)
    
    for i in range(N):
        # get mask regions
        mask_region = attribution < max_attribution - i * interval
        masked_input = original_input.clone()
        # use blurred_original_input to mask the input
        masked_input = torch.where(~mask_region, masked_input, blurred_original_input)
        
        # mean target score of the batch
        score, acc = forward(masked_input, target_label)
        y_axis.append(score)
        
    # plot and save the curve
    x_axis = np.arange(0, 1+step, step)
    y_axis = np.array(y_axis)
    auc = np.trapz(y_axis, x_axis)
    
    # plot
    # import matplotlib.pyplot as plt
    # plt.plot(x_axis, y_axis)
    
    return auc, x_axis, y_axis


def knm(original_input, attribution, target_label, forward, step=0.05):
    """ knm: keep negative mask 
        works like kpm, but we keep the most low attribution pixels and mask other pixels,
        and increase the numbers of input pixels that attribution is negative from low attribution to high attribution, other pixels are masked using the average value of masked pixels
    """
    
    B, C, H, W = original_input.size()
    # get sampling interval
    max_attribution = 0. # attribution.max()
    min_attribution = attribution.min()
    interval = (max_attribution - min_attribution) * step
    N = int(1 / step) + 1
    
    y_axis = []
    # get blurred input
    blurred_original_input = GaussianBlur(51, 41)(original_input)
    for i in range(N):
        # get mask regions
        mask_region = attribution > min_attribution + i * interval
        masked_input = original_input.clone()
        # use blurred_original_input to mask the input
        masked_input = torch.where(~mask_region, masked_input, blurred_original_input)
        
        # mean target score of the batch
        score, acc = forward(masked_input, target_label)
        y_axis.append(score)
        
    x_axis = np.arange(0, 1+step, step)
    y_axis = np.array(y_axis)
    auc = np.trapz(y_axis, x_axis)
    
    return auc, x_axis, y_axis


def kam(original_input, attribution, target_label, forward, step=0.05):
    """ kam: keep absolute mask
        works like kpm, but we now observe the accuracy of the predictions instead of the prediction score of target label
        also use absolute attritbiton map instead of positive attribution map
    """
    
    B, C, H, W = original_input.size()
    attribution = attribution.abs()
    # get sampling interval
    max_attribution = attribution.max()
    min_attribution = attribution.min()
    interval = (max_attribution - min_attribution) * step
    N = int(1 / step) + 1
    
    y_axis = []
    blurred_original_input = GaussianBlur(51, 41)(original_input)
    for i in range(N):
        # get mask regions
        mask_region = attribution < max_attribution - i * interval
        masked_input = original_input.clone()
        # use blurred_original_input to mask the input
        masked_input = torch.where(~mask_region, masked_input, blurred_original_input)
        
        # mean acc of the batch
        score, acc = forward(masked_input, target_label)
        y_axis.append(acc)
        
    x_axis = np.arange(0, 1+step, step)
    y_axis = np.array(y_axis)
    auc = np.trapz(y_axis, x_axis)
    
    return auc, x_axis, y_axis


def ram(original_input, attribution, target_label, forward, step=0.05):
    """ ram: remove absolute mask
        works like knm, but we now observe the accuracy of the predictions instead of the prediction score of target label
        also use absolute attritbiton map instead of positive attribution map
    """
    
    B, C, H, W = original_input.size()
    attribution = attribution.abs()
    # get sampling interval
    max_attribution = attribution.max()
    min_attribution = attribution.min()
    interval = (max_attribution - min_attribution) * step
    N = int(1 / step) + 1
    
    y_axis = []
    blurred_original_input = GaussianBlur(51, 41)(original_input)
    for i in range(N):
        # get mask regions
        mask_region = attribution > max_attribution - i * interval
        masked_input = original_input.clone()
        # use blurred_original_input to mask the input
        masked_input = torch.where(~mask_region, masked_input, blurred_original_input)
        
        # mean acc of the batch
        score, acc = forward(masked_input, target_label)
        y_axis.append(acc)
        
    x_axis = np.arange(0, 1+step, step)
    y_axis = np.array(y_axis)
    auc = np.trapz(y_axis, x_axis)
    
    return auc, x_axis, y_axis
