import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.models.quantization
import copy
import os
import yaml
import json
import torch.ao.quantization

def quantize_model(model_weights_path, config, summary_metric):
    device = torch.device('cpu')
    
