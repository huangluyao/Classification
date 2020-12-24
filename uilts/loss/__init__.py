from .loss import FocalLoss
import torch.nn as nn


def get_loss(cfg):
    return {
        "Focal_Loss": FocalLoss(cfg["n_classes"]),
        "CrossEntropyLoss": nn.CrossEntropyLoss()
    }[cfg["loss"]]