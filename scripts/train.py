# generic train entry point script for any algorithm
import argparse
import wandb
import yaml
import logging
import sys
import torch
import os
import datetime

from pathlib import Path
from argparse import Namespace
from typing import Optional, Sequence, Dict, Any
from distutils.util import strtobool
from common.utils import setup_logging, create_instance_from_spec, print_spec, parse_spec_overrides


logger = logging.getLogger(__name__)


def parse_command_line(args: Optional[Sequence[str]] = None) -> Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument('--experiment_spec_file', type=str, required=True, help='Path to spec yaml file for algorithm')
    parser.add_argument('--spec_overrides', nargs=argparse.REMAINDER, default=None, help='Overrides to the spec yaml file')
    parser.add_argument('--results_dir', type=str, default='results')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--load_from_checkpoint', type=str, default=None)
    # wandb config params
    parser.add_argument('--use_wandb', type=lambda x: bool(strtobool(x)), default=False)
    parser.add_argument('--wandb_entity', type=str, default='qdrl')
    parser.add_argument('--wandb_project', type=str, default='alda')
    parser.add_argument('--wandb_group', type=str, default='alda')
    parser.add_argument('--wandb_run_name', type=str, default='alda_dmcgb')
    parser.add_argument('--wandb_tag', type=str, default='dmcgb')

    return parser.parse_args(args)


def parse_spec_file(args: Namespace) -> Dict[str, Any]:
    with open(args.experiment_spec_file, 'r', encoding='utf-8') as src:
        spec = yaml.load(src, Loader=yaml.loader.FullLoader)

    if args.spec_overrides is not None:
        spec = parse_spec_overrides(spec, args.spec_overrides)

    if spec['name'] in ["grid", "datetime"]:
        # set name as current date time
        spec['name'] = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    # set device
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    spec['trainer']['config']['device'] = device

    spec['trainer']['config']['use_wandb'] = args.use_wandb
    spec['trainer']['config']['debug'] = args.debug

    seed = spec['trainer']['config']['seed']
    exp_dir = Path(args.results_dir).joinpath(spec['name']).joinpath(f'seed_{seed}') if not args.debug else Path(args.results_dir).joinpath('debug')
    exist_ok = False if not args.debug else True
    exist_ok = exist_ok or args.load_from_checkpoint is not None
    os.makedirs(exp_dir, exist_ok=exist_ok)
    spec['trainer']['config']['exp_dir'] = str(exp_dir)

    return spec


def _setup_logging(spec: Dict[str, Any]) -> None:
    """Sets up logging."""

    log_dir = Path(spec["trainer"]["config"]["exp_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "output_log_0.txt"

    debug_mode = spec["trainer"]["config"]["debug"]
    setup_logging(
        log_level=spec.get("log_level", logging.DEBUG if debug_mode else logging.INFO),
        log_file=str(log_file),
    )


def _setup_wandb(args: Dict[str, Any], spec: Dict[str, Any]) -> None:
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


def main(cl_args: Optional[Sequence[str]] = None) -> None:
    args = parse_command_line(cl_args)

    spec = parse_spec_file(args)

    _setup_logging(spec)

    args = vars(args)
    if args['use_wandb']:
        _setup_wandb(args, spec)

    # torch.multiprocessing.set_start_method("spawn")

    # Log command line, some system stats and final spec.
    logger.info(" ".join(sys.argv))

    print_spec(spec, 0)

    trainer = create_instance_from_spec(spec['trainer'], name=spec['name'])

    # record final spec after overrides
    log_dir = Path(spec['trainer']['config']['exp_dir'])
    with open(log_dir / 'experiment_spec_final.yaml', 'w', encoding='utf-8') as w:
        yaml.safe_dump(spec, w)

    trainer.build(spec)

    if args['load_from_checkpoint']:
        logger.info(f"Loading checkpoint from {args['load_from_checkpoint']}")
        trainer.load_checkpoint(args['load_from_checkpoint'])

    trainer.train()

    logger.info('All done')


if __name__ == '__main__':
    main()
