from .evalute import *

def get_evalution(model, loader, criteon, cfg):
    return {
        "AccScore": AccScore(model, loader, criteon, cfg),
        "F1Score": F1Score(model, loader, criteon, cfg)
        }[cfg["evaluation"]]


