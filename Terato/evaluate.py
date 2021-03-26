import copy
import csv
import os
import time

import numpy as np
import torch
from tqdm import tqdm

def evaluate_sample(masks_true, masks_pred, masks_names, metrics):
    metrics_results = {}
    for i, mask_name in enumerate(masks_names):
        mask_pred = torch.split(masks_pred, 1, dim = 1)[i].data.numpy().ravel()
        mask_true = torch.split(masks_true, 1, dim = 1)[i].data.numpy().ravel()
        if sum(mask_true != 0):
            for name, metric in metrics.items():
                if name == 'f1_score':
                    # Use a classification threshold of 0.1
                    metrics_results[f'{name}_{mask_name}'] = metric(mask_true > 0, mask_pred > 0.1, zero_division = 1)
                else:
                    metrics_results[f'{name}_{mask_name}'] = metric(mask_true.astype('uint8'), mask_pred)
    return metrics_results


def test_evaluation(model, criterion, dataloaders, metrics, masks_names, bpath, device='cpu'):
    since = time.time()

    if device != 'cpu':
        if torch.cuda.is_available():
            device = 'cuda:0'
            print('Device set correctly to cuda.')
        else:
            print('Cuda device is not available. Device set to cpu.')
    device = torch.device(device)
    model.to(device)
    model.eval()  # Set model to evaluate mode

    metrics_global = {f'{m}_{mask_name}':[] for m in metrics.keys() for mask_name in masks_names}
    metrics_global['Loss'] = []
    for batch in tqdm(iter(dataloaders['Test'])):
        for input,masks in zip(torch.split(batch['image'], 1, dim = 0), torch.split(batch['masks'],1,dim = 0)):
            print(masks.shape)
            input = input.to(device)
            masks = masks.to(device)
            # We don't care about the gradients when evaluating
            with torch.set_grad_enabled(False):
                outputs = model(input)
                metrics_sample = evaluate_sample(masks,outputs['out'],masks_names,metrics)
                for metric_name, metric_value in metrics_sample.items():
                    metrics_global[metric_name].append(metric_value)
                metrics_global['Loss'].append(criterion(outputs['out'], masks))

    for field in metrics_global:
        metrics_global[field] = np.mean(metrics_global[field])
    return metrics_global
