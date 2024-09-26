from torchmetrics.functional.regression import spearman_corrcoef
import torch

@torch.no_grad()
def spearman_rank_correlation_value(att1:torch.Tensor, att2: torch.Tensor):
    assert att1.size() == att2.size(), 'Attribution maps must have the same size'
    att1 = att1.view(att1.size(0), -1)
    att2 = att2.view(att2.size(0), -1)
    corr = spearman_corrcoef(att1.T, att2.T)
    return corr.cpu()

def class_specific_spearman_rank_corr(att_target, att_other_class_list):
    '''
    Return: 
        corr: torch.Tensor, shape (Batch_size,)
    '''
    corr = torch.zeros(att_target.size(0))
    for att_other_class in att_other_class_list:
        class_corr = spearman_rank_correlation_value(att_target, att_other_class)
        corr += class_corr
    return corr / len(att_other_class_list)

if __name__ == '__main__':
    device = torch.device('cuda')
    att1 = torch.rand(32, 1, 224, 224).to(device)
    att2 = -att1.clone().to(device)
    att3 = att1.clone().to(device)
    att4 = torch.rand(32, 1, 224, 224).to(device)
    att4[1] = -att1[1]

    att_other_class_list = [att2, att3, att4]
    corr = class_specific_spearman_rank_corr(att1, att_other_class_list)
    print('\n', corr)