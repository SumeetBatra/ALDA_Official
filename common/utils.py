import json
import torch
import logging
import importlib
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as td
import numpy as np
import sapien.core as sapien
import imageio
import os
import wandb

from distutils.util import strtobool
from sapien.core import Pose
from torch.nn.utils import spectral_norm
from colorlog import ColoredFormatter
from typing import Dict, Any, List
from transforms3d.quaternions import mat2quat

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


def calc_num_elements(module, module_input_shape):
    shape_with_batch_dim = (1,) + module_input_shape
    some_input = torch.rand(shape_with_batch_dim)
    num_elements = module(some_input).numel()
    return num_elements


def fc_layer(in_features: int, out_features: int, bias=True, spec_norm=False) -> nn.Module:
    layer = nn.Linear(in_features, out_features, bias)
    if spec_norm:
        layer = spectral_norm(layer)

    return layer


def create_mlp(layer_sizes: List[int], input_size: int, activation: nn.Module) -> nn.Module:
    """Sequential fully connected layers."""
    layers = []
    for i, size in enumerate(layer_sizes):
        layers.extend([fc_layer(input_size, size), activation])
        input_size = size

    if len(layers) > 0:
        return nn.Sequential(*layers)
    else:
        return nn.Identity()


def normalize(tensor: torch.Tensor) -> torch.Tensor:
    # normalize rgbd tensor from maniskills to [-1, 1]. rgb is already [0,1], while depth is [0, 3.7257]
    tensor[3].div_(3.7257)
    tensor.mul_(2.).sub_(1.)
    return tensor


def denormalize(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.clamp_(-1, 1).add_(1).div_(2).mul_(255)


#  https://github.com/kc-ml2/SimpleDreamer/blob/master/dreamer/modules/model.py
def initialize_weights(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.kaiming_uniform_(m.weight.data, nonlinearity="relu")
        nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight.data)
        nn.init.constant_(m.bias.data, 0)


def horizontal_forward(network, x, y=None, input_shape=(-1,), output_shape=(-1,)):
    batch_with_horizon_shape = x.shape[: -len(input_shape)]
    if not batch_with_horizon_shape:
        batch_with_horizon_shape = (1,)
    if y is not None:
        x = torch.cat((x, y), -1)
        input_shape = (x.shape[-1],)  #
    x = x.reshape(-1, *input_shape)
    x = network(x)

    x = x.reshape(*batch_with_horizon_shape, *output_shape)
    return x


def create_normal_dist(
        x,
        std=None,
        mean_scale=1,
        init_std=0,
        min_std=0.1,
        activation=None,
        event_shape=None,
):
    if std == None:
        mean, std = torch.chunk(x, 2, -1)
        mean = mean / mean_scale
        if activation:
            mean = activation(mean)
        mean = mean_scale * mean
        std = F.softplus(std + init_std) + min_std
    else:
        mean = x
    dist = torch.distributions.Normal(mean, std)
    if event_shape:
        dist = torch.distributions.Independent(dist, event_shape)
    return dist


def create_categorical_dist(logits):
    return td.Independent(td.OneHotCategoricalStraightThrough(logits=logits), 1)


def build_network(input_size, hidden_size, num_layers, activation, output_size):
    assert num_layers >= 2, "num_layers must be at least 2"
    activation = getattr(nn, activation)()
    layers = []
    layers.append(nn.Linear(input_size, hidden_size))
    layers.append(activation)

    for i in range(num_layers - 2):
        layers.append(nn.Linear(hidden_size, hidden_size))
        layers.append(activation)

    layers.append(nn.Linear(hidden_size, output_size))

    network = nn.Sequential(*layers)
    # network.apply(initialize_weights)
    return network


def bottle(f, xs):
    # Wraps the input for a function to process a (time, batch, feature) sequence in (time * batch, feature)
    horizon, batch_size = xs[0].shape[:2]
    ys = f(*(x.reshape(horizon * batch_size, *x.shape[2:]) for x in xs))
    if isinstance(ys, tuple):
        return tuple(y.reshape(horizon, batch_size, *y.shape[1:]) for y in ys)
    else:
        return ys.reshape(horizon, batch_size, *ys.shape[1:])


def preprocess(obs):
    # Preprocess a batch of observations
    ndims = len(obs.shape)
    assert ndims == 2 or ndims == 4, "preprocess accepts a batch of observations"
    if ndims == 4:
        obs = obs / 255.
    return obs


_GPU_ID = 0
_USE_GPU = True
_DEVICE = 'cuda'


def get_device():
    global _DEVICE
    return _DEVICE


def to_torch(x, dtype=None, device=None):
    if device is None:
        device = get_device()
    return torch.as_tensor(x, dtype=dtype, device=device)


def to_np(x):
    return x.detach().cpu().numpy()


class FreezeParameters:
    def __init__(self, params):
        self.params = params
        self.param_states = [p.requires_grad for p in self.params]

    def __enter__(self):
        for param in self.params:
            param.requires_grad = False

    def __exit__(self, exc_type, exc_val, exc_tb):
        for i, param in enumerate(self.params):
            param.requires_grad = self.param_states[i]


def normalize_vector(x, eps=1e-6):
    x = np.asarray(x)
    assert x.ndim == 1, x.ndim
    norm = np.linalg.norm(x)
    if norm < eps:
        return np.zeros_like(x)
    else:
        return x / norm


def look_at(eye, target, up=(0, 0, 1)):
    """Get the camera pose in SAPIEN by the Look-At method.

    Note:
        https://www.scratchapixel.com/lessons/mathematics-physics-for-computer-graphics/lookat-function
        The SAPIEN camera follows the convention: (forward, right, up) = (x, -y, z)
        while the OpenGL camera follows (forward, right, up) = (-z, x, y)
        Note that the camera coordinate system (OpenGL) is left-hand.

    Args:
        eye: camera location
        target: looking-at location
        up: a general direction of "up" from the camera.

    Returns:
        sapien.Pose: camera pose
    """
    forward = normalize_vector(np.array(target) - np.array(eye))
    up = normalize_vector(up)
    left = np.cross(up, forward)
    up = np.cross(forward, left)
    rotation = np.stack([forward, left, up], axis=1)
    return sapien.Pose(p=eye, q=mat2quat(rotation))


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

