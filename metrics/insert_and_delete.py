'''
adapted from https://github.com/LMBTough/MFABA
@article{zhu2023mfaba,
  title={MFABA: A More Faithful and Accelerated Boundary-based Attribution Method for Deep Neural Networks},
  author={Zhu, Zhiyu and Chen, Huaming and Zhang, Jiayu and Wang, Xinyi and Jin, Zhibo and Xue, Minhui and Zhu, Dongxiao and Choo, Kim-Kwang Raymond},
  journal={arXiv preprint arXiv:2312.13630},
  year={2023}
}
'''

import torch
import numpy as np

@torch.no_grad()
def insert_score(model, step, images, explanations):
    model.eval()
    B, C, H, W = images.size()
    if explanations.size(1) == 1:
        explanations = explanations.expand(-1, C, -1, -1)
    predictions = model(images)
    top, c = torch.max(predictions, -1)
    n_steps = (H * W + step - 1) // step
    scores = np.empty((B, n_steps + 1))
    salient_order = explanations.view(B, C, H * W).argsort(descending=True)
    
    start = torch.zeros_like(images, device=images.device)
    finish = images.clone()
    finish = finish.view(B, C, H * W)

    for i in range(n_steps + 1):
        pred = model(start)
        pred = torch.nn.functional.softmax(pred, dim=-1)
        scores[:, i] = pred[torch.arange(B), c].cpu().numpy()
        if i < n_steps:
            coords = salient_order[:, :, i * step:(i + 1) * step]
            # change the value of the pixels according to the coords
            start = start.view(B, C, H * W)
            start.scatter_(dim=2, index=coords, src=torch.gather(finish, dim=2, index=coords))
            start = start.view(B, C, H, W)
            
    scores = np.sum(scores, axis=0)
    xs = np.linspace(0, 1, scores.shape[0])
    auc = np.trapz(scores, dx=xs[1] - xs[0]) / B

    return auc, scores

@torch.no_grad()
def delete_score(model, step, images, explanations):
    model.eval()
    B, C, H, W = images.size()
    if explanations.size(1) == 1:
        explanations = explanations.expand(-1, C, -1, -1)
    predictions = model(images)
    top, c = torch.max(predictions, -1)
    n_steps = (H * W + step - 1) // step
    scores = np.empty((B, n_steps + 1))
    salient_order = explanations.view(B, C, H * W).argsort(descending=True)
    
    start = images.clone()
    finish = torch.zeros_like(images, device=images.device)
    finish = finish.view(B, C, H * W)

    for i in range(n_steps + 1):
        pred = model(start)
        pred = torch.nn.functional.softmax(pred, dim=-1)
        scores[:, i] = pred[torch.arange(B), c].cpu().numpy()
        if i < n_steps:
            coords = salient_order[:, :, i * step:(i + 1) * step]
            # change the value of the pixels according to the coords
            start = start.view(B, C, H * W)
            start.scatter_(dim=2, index=coords, src=torch.gather(finish, dim=2, index=coords))
            start = start.view(B, C, H, W)

    scores = np.sum(scores, axis=0)
    xs = np.linspace(0, 1, scores.shape[0])
    auc = np.trapz(scores, dx=xs[1] - xs[0]) / B
    return auc, scores

class Insert_Delete_Metric():

    def __init__(self, model, mode, step):
        r"""Create deletion/insertion metric instance.
        Args:
            model (nn.Module): Black-box model being explained.
            mode (str): 'del' or 'ins'.
            step (int): number of pixels modified per one iteration.
            baseline (str): 'black' or 'white'.
        """
        assert mode in ['del', 'ins']
        self.model = model
        self.mode = mode
        self.step = step
        self.substrate_fn = lambda x: torch.zeros_like(x, device=x.device)

    def single_run(self, image, explanation: torch.Tensor):
        ''' only one image
        image: torch.Tensor, 1, C, H, W
        explanation: torch.Tensor, 1, 1, H, W or 1, C, H, W
        '''
        _, C, H, W = image.size()
        if explanation.size(1) == 1:
            explanation = explanation.expand(-1, C, -1, -1)
        pred = self.model(image)
        top, c = torch.max(pred, 1)
        n_steps = (H * W + self.step - 1) // self.step

        if self.mode == 'del':
            start = image.clone()
            finish = self.substrate_fn(image)
        else:
            start = self.substrate_fn(image)
            finish = image.clone()
        finish = finish.view(-1, C, H * W)

        scores = np.empty(n_steps + 1)
        salient_order = explanation.view(-1, H * W).argsort(descending=True)
        for i in range(n_steps+1):
            pred = self.model(start)
            pred = torch.nn.functional.softmax(pred, dim=-1)
            scores[i] = pred[0, c]
            if i < n_steps:
                coords = salient_order[:, i * self.step:(i + 1) * self.step]
                start = start.view(-1, C, H * W)
                start[:, :, coords] = finish[:, :, coords]
                start = start.view(-1, C, H, W)
        auc = np.trapz(scores, dx=self.step)
        return auc, scores
    
    def batch_run(self, images, explanations):
        B, C, H, W = images.size()
        if explanations.size(1) == 1:
            explanations = explanations.expand(-1, C, -1, -1)
        predictions = self.model(images)
        top, c = torch.max(predictions, -1)
        n_steps = (H * W + self.step - 1) // self.step
        scores = np.empty((B, n_steps + 1))
        salient_order = explanations.view(B, C, H * W).argsort(descending=True)
        if self.mode == 'del':
            start = images.clone()
            finish = self.substrate_fn(images)
        else:
            start = self.substrate_fn(images)
            finish = images.clone()

        for i in range(n_steps + 1):
            pred = self.model(start)
            pred = torch.nn.functional.softmax(pred, dim=-1)
            scores[:, i] = pred[torch.arange(B), c]
            if i < n_steps:
                coords = salient_order[:, :, i * self.step:(i + 1) * self.step]
                start = start.view(B, C, H * W)
                start[torch.arange(B).unsqueeze(1), :, coords] = finish[torch.arange(B).unsqueeze(1), :, coords]
                start = start.view(B, C, H, W)
        
        auc = np.trapz(scores, dx=self.step, axis=1)

        return auc, scores