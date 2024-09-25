import torch
from torchvision.transforms import GaussianBlur
import numpy as np

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
    blurred_original_input = GaussianBlur(51, 41)(original_input)
    
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