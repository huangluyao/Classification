from .DenseNet import DenseNet
from .mobilenetv3 import mobilenetv3_large
from .Xception import xception
from .MnasNet import mnasnet1_3
from .EfficientNet import efficientnet_b6


def get_model(model_name, cfg):
    model = {"MobileNetv3": mobilenetv3_large,
             "Xception": xception,
             "MobilNetv3": mnasnet1_3,
             "EfficientNet": efficientnet_b6
             }[model_name](cfg["n_classes"])

    if "pre_trian_model" in cfg.keys():
        import torch
        model.load_state_dict(torch.load(cfg["pre_trian_model"]))
    return model
