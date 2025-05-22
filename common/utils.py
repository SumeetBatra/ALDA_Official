import json
import logging
import importlib
import torch.nn as nn
import numpy as np
import imageio
import os
import wandb

from distutils.util import strtobool
from colorlog import ColoredFormatter
from typing import Dict, Any, List

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s", level=logging.INFO
)
logger = logging.getLogger('utils')

grad_magnitudes = {}


def setup_wandb(args: Dict[str, Any], spec: Dict[str, Any]) -> None:
    """Sets up wandb experiment tracking if enabled"""
    run_name = args['wandb_run_name'] + f"_seed_{spec['trainer']['config']['seed']}"
    wandb.init(
        project=args['wandb_project'],
        entity=args['wandb_entity'],
        group=args['wandb_group'],
        name=run_name,
        tags=[args['wandb_tag']],
        config=spec
    )


class eval_mode(object):
    def __init__(self, *models):
        self.models = models

    def __enter__(self):
        self.prev_states = []
        for model in self.models:
            self.prev_states.append(model.training)
            model.train_mode(False)

    def __exit__(self, *args):
        for model, state in zip(self.models, self.prev_states):
            model.train_mode(state)
        return False


class VideoRecorder:
    def __init__(self, dir_name, height=448, width=448, camera_id=0, fps=25):
        self.dir_name = dir_name
        self.height = height
        self.width = width
        self.camera_id = camera_id
        self.fps = fps
        self.frames = []

    def init(self, enabled=True):
        self.frames = []
        self.enabled = self.dir_name is not None and enabled

    def record(self, env, mode=None, frame=None):
        if self.enabled:
            if frame is None:
                frame = env.render(
                    mode='rgb_array',
                    height=self.height,
                    width=self.width,
                    camera_id=self.camera_id
                )
            else:
                rendered_frame = env.render(
                    mode='rgb_array',
                    height=self.height,
                    width=self.width // 2,
                    camera_id=self.camera_id
                )
                frame = np.concatenate((frame[..., -3:], rendered_frame), axis=1)
            if mode is not None and 'video' in mode:
                _env = env
                while 'video' not in _env.__class__.__name__.lower():
                    _env = _env.env
                frame = _env.apply_to(frame)
            self.frames.append(frame)

    def save(self, file_name):
        if self.enabled:
            path = os.path.join(self.dir_name, file_name)
            imageio.mimsave(path, self.frames, fps=self.fps)


def create_instance_from_spec(spec: Dict[str, Any], *args, **kwargs) -> Any:
    '''
    Creates instance of a class from a spec dict loaded from a yaml file
    :param spec: spec dict loaded from yaml file
    '''
    module_name = spec['module']
    module = importlib.import_module(module_name)
    class_obj = getattr(module, spec['class'])
    return class_obj(*args, **spec['config'], **kwargs)


def print_spec(spec, indent):
    """Pretty prints spec file."""
    if indent == 0:
        logger.info("---- Options ----")
    # Print spec.
    max_len = max([len(k) for k in spec.keys()])
    fmt_string = " " * indent + "{{:<{}}}: {{}}".format(max_len)
    dicts = []
    for k, v in sorted(spec.items()):
        if type(v) is dict:
            dicts.append((k, v))
        else:
            logger.info(fmt_string.format(str(k), str(v)))
    for k, v in dicts:
        logger.info(" " * indent + k + ":")
        if v:
            print_spec(v, indent + 4)
    if indent == 0:
        logger.info("-----------------")


def setup_logging(log_level: int = logging.INFO, log_file: str = None):
    """Sets up logging for scripts."""
    ch = logging.StreamHandler()
    ch.setLevel(log_level)

    formatter = ColoredFormatter(
        "%(log_color)s[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
        datefmt=None,
        reset=True,
        log_colors={
            'DEBUG': 'cyan',
            'INFO': 'white,bold',
            'INFOV': 'cyan,bold',
            'WARNING': 'yellow',
            'ERROR': 'red,bold',
            'CRITICAL': 'red,bg_white',
        },
        secondary_log_colors={},
        style='%'
    )
    ch.setFormatter(formatter)

    # configure streaming to logfile
    if log_file is not None:
        fh = logging.FileHandler(log_file)
        fh.setLevel(log_level)
        fh.setFormatter(formatter)

    # configure the console stream
    logging.root.addHandler(ch)



#  https://github.com/kc-ml2/SimpleDreamer/blob/master/dreamer/modules/model.py
def initialize_weights(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.kaiming_uniform_(m.weight.data, nonlinearity="relu")
        nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight.data)
        nn.init.constant_(m.bias.data, 0)


def parse_spec_overrides(spec, other_args, allow_new_keys=False):
    """Handles spec file overrides.

    Parses optional spec file overrides and updates provided spec container.
    Spec overrides are passed as command line arguments with 'spec.' prefix .
    This must be done before any post-processing happens so the user can expect
    consistent behavior as if the spec file was changed by the user before running any code.

    Args:
        spec: spec dictionary.
        other_args: list of extra arguments returned by argparse.parse_known_args.
        allow_new_keys: allow sections and keys that don't exist in `spec`, note that new value
            literals are parsed by JSON lexer.

    Returns updated spec. Note that the spec is updated in-place.
    """

    # 1. Convert other_args to dictionary.
    # This is a very primitive parser, be gentle with it...
    spec_overrides = {}
    for a in other_args:
        items = a.split("=")
        if len(items) != 2:
            raise ValueError(
                "Invalid command line argument: {}. Expected key=value".format(a)
            )
        k, v = items
        if not k.startswith("--spec."):
            raise ValueError(
                'Spec override key must start with "--spec." but got: {}'.format(k)
            )
        k = k[len("--spec."):].strip()
        spec_overrides[k] = v.strip()

    # 2. Handle spec overrides, if any.
    for k, v in spec_overrides.items():
        section_keys = k.split(".")
        section = spec
        try:
            for skey in section_keys[:-1]:
                if skey not in section and allow_new_keys:
                    section[skey] = {}
                section = section[skey]
            if section_keys[-1] not in section and allow_new_keys:
                # Use JSON lexer to determine the type and value of the new key-value literal.
                try:
                    new_val = json.loads(v)
                except json.JSONDecodeError:
                    new_val = json.loads(f'"{v}"')
            else:
                cur_val = section[section_keys[-1]]
                cur_val_t = type(cur_val)
                # Return strings as-is, use JSON parser otherwise.
                # This is useful in case of complex overrides, such as lists.
                # Need to use string_types to cover str/unicode in Py2/Py3.
                new_val = v if isinstance(cur_val, str) or isinstance(cur_val, bool) else cur_val_t(json.loads(v))
                if isinstance(cur_val, bool):
                    new_val = bool(strtobool(new_val))
            section[section_keys[-1]] = new_val
        except KeyError:
            logger.error("Spec override key not found: {}".format(k))
            raise
        except ValueError:
            logger.error("Spec override {} value {} has invalid type.".format(k, v))
            raise
    return spec


def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


class GradMagnitudeTracker:
    def __init__(self, model):
        self.model = model
        self.grad_magnitudes = {}

        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.register_hook(self.create_hook(name))

    def create_hook(self, name):
        def hook(grad):
            mag = grad.norm().item()
            self.grad_magnitudes[name] = mag
        return hook

    def clear(self):
        self.grad_magnitudes.clear()

    def get_grad_magnitudes(self):
        return self.grad_magnitudes

