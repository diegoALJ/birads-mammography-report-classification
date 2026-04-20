import os
import random
from types import SimpleNamespace

import numpy as np
import torch
import yaml
from sklearn.metrics import f1_score


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_score(y_true, y_pred):
    return f1_score(y_true, y_pred, average="macro")


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.sum = 0.0
        self.count = 0
        self.avg = 0.0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0.0


def load_yaml_config(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def dict_to_namespace(d):
    if isinstance(d, dict):
        return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in d.items()})
    if isinstance(d, list):
        return [dict_to_namespace(x) for x in d]
    return d


def prepare_output_dirs(cfg) -> None:
    os.makedirs(cfg.paths.output_dir, exist_ok=True)
    os.makedirs(cfg.paths.model_dir, exist_ok=True)
    os.makedirs(cfg.paths.oof_dir, exist_ok=True)
    os.makedirs(cfg.paths.submission_dir, exist_ok=True)
    os.makedirs(cfg.paths.cache_dir, exist_ok=True)
