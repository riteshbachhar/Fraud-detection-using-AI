#!/usr/bin/env python3
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F

# loading config file
def load_config(config_path):
    """ Load configuration from a YAML file. """
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

# Focal Loss
class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance - better than BCE for F2 optimization"""
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()
