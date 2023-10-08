import torch

import os

def get_device(device_idx=0):
    """
    Set device to cuda if available, else cpu
    Input:
        device_idx: int, index of cuda device to use
    Output:
        device: torch.device, device to use
    """
    if torch.cuda.is_available():  # cuda:0 or cuda:1 for our 2-GPU testbed
        device = torch.device("cuda:" + str(device_idx))
    else:
        device = torch.device("cpu")
        
    return device

def save_results(results, save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    outputs_save_path = os.path.join(save_path, "outputs.pt")
    labels_save_path = os.path.join(save_path, "labels.pt")
    torch.save(results["outputs"], outputs_save_path)
    torch.save(results["labels"], labels_save_path)