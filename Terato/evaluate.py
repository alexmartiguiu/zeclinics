import copy
import csv
import os
import time

import numpy as np
import torch
import time
from tqdm import tqdm

def evaluate_sample(masks_true, masks_pred, masks_names, metrics):
    #ini = time.time()
    metrics_results = {}
    masks_pred = torch.split(masks_pred, 1, dim = 1)
    masks_true = torch.split(masks_true, 1, dim = 1)
    for i, mask_name in enumerate(masks_names):
        #if i == 0: print("begin splits",time.time()-ini)
        mask_pred = masks_pred[i].cpu().data.numpy().ravel()
        mask_true = masks_true[i].cpu().data.numpy().ravel()
        #if i == 0: print("end splits", time.time()-ini)
        if mask_true.any() != 0:
            #if i == 0: print("sum finished", time.time()-ini)
            for name, metric in metrics.items():
                if name in ['f1_score','precision','recall']:
                    # Use a classification threshold of 0.1
                    metrics_results[f'{name}_{mask_name}'] = metric(mask_true > 0, mask_pred > 0.01)
                else:
                    metrics_results[f'{name}_{mask_name}'] = metric(mask_true.astype('uint8'), mask_pred)
        #if i == 0: print("end iteration",time.time()-ini)
    return metrics_results


def test_evaluation(model, criterion, dataloaders, metrics, masks_names, device='cpu'):
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
