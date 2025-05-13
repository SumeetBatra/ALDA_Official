import torch
import torch.nn as nn
import os
import random
import numpy as np

from pathlib import Path
from typing import Any, Dict
from abc import abstractmethod

from torch.cuda.amp import GradScaler

from common.utils import create_instance_from_spec as from_spec


class TrainerBase:
    name: str

    seed: int

    exp_dir: str

    checkpoint_n_epochs: int
    validate_n_epochs: int

    deterministic: bool = False

    use_wandb: bool = False

    debug: bool = False

    grad_clip: bool = False

    train_batch_size: int

    start_epoch: int = 1
    num_epochs: int

    model: nn.Module
    optimizer: torch.optim.Optimizer

    device: str

    amp: bool = False

    def __init__(self, **kwargs):
        TrainerBase.set_attributes(self, kwargs)
        self.spec = None
        self.cp_dir = Path(os.path.join(self.exp_dir, 'checkpoints'))
        self.cp_dir.mkdir(exist_ok=True)

        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
        random.seed(self.seed)
        np.random.seed(self.seed)

        # torch.backends.cudnn.benchmark = not self.deterministic
        # torch.backends.cudnn.deterministic = self.deterministic

        self.scaler = None
        if self.amp:
            self.scaler = torch.cuda.amp.GradScaler()

    def build(self, spec: Dict[str, Any]) -> None:
        self.spec = spec

    @classmethod
    def set_attributes(cls, obj: Any, values: Dict[str, Any]) -> None:
        """Uses annotations to set the attributes of the instance object."""
        ann = vars(cls).get("__annotations__")
        if not isinstance(ann, dict):
            return
        for name in ann.keys():
            # if (value := values.pop(name, None)) is not None:
            #     setattr(obj, name, value)
            value = values.pop(name, None)
            if value is not None:
                setattr(obj, name, value)

    @abstractmethod
    def save_checkpoint(self, epoch: int):
        return NotImplementedError

    @abstractmethod
    def load_checkpoint(self, path: str):
        return NotImplementedError

    @abstractmethod
    def train(self):
        raise NotImplementedError
