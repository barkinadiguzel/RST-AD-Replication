import torch
import torch.nn.functional as F
import numpy as np

def generate_anomaly(img, mask=None):
    anomaly = torch.rand_like(img) * 0.2  
    if mask is not None:
        anomaly = anomaly * mask
    blended = torch.clamp(img + anomaly, 0, 1)
    return blended
