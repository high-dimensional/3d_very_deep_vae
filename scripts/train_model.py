"""Script to train variational autoencoder model on neuroimaging data"""

import argparse
import json
import os
from pathlib import Path
from typing import Any
import torch
from verydeepvae.orchestration import training_script_vae_new as training_script


def parse_command_line_arguments() -> argparse.Namespace:
    """Parse command line arguments."""    
    parser = argparse.ArgumentParser(
        "Train variational autoencoder model on neuroimaging data"
    )
    parser.add_argument(
        "--json_config_file", 
        type=Path,
        required=True,
        help="Path to JSON file specifying model / run hyperparameters"
    )
    parser.add_argument(
        "--nifti_flair_dir", 
        type=Path,
        required=True,
        help="Path to directory containing FLAIR image NiFTI files to train model with"
    )
    parser.add_argument(
        "--output_dir", 
        type=Path, 
        required=True,
        help="Directory to save run outputs to"
    )
    parser.add_argument(
        "--local_rank", 
        type=int, 
        default=0, 
        help="Rank of node when running on multiple nodes in parallel"
    )
    parser.add_argument(
        "--CUDA_devices", 
        type=str,
        nargs="+",
        default=["0"],
        help="Device indices (zero-based) for GPUs to use when training model"
    )
    return parser.parse_args()


def post_process_hyperparameters(hyperparameters: dict[str, Any], cli_args: argparse.Namespace):
    """Update loaded hyperparameter dictionary in-place using CLI argument values."""
    hyperparameters["model_name"] = cli_args.json_config_file.stem
    hyperparameters['checkpoint_folder'] = str(cli_args.output_dir / 'torch_checkpoints')
    hyperparameters['tensorboard_dir'] = str(cli_args.output_dir / 'tensorboard')
    hyperparameters['recon_folder'] = str(cli_args.output_dir / 'reconstructions')
    for key in ["nifti_flair_dir", "CUDA_devices", "local_rank"]:
        hyperparameters[key] = getattr(cli_args, key)
    if (
        not isinstance(hyperparameters["latents_per_channel_weight_sharing"], list) 
        and hyperparameters["latents_per_channel_weight_sharing"] == "none"
    ):
        assert "latents_per_channel" in hyperparameters, (
            "latents_per_channel must be specified in configuration file "
            "if latents_per_channel_weight_sharing=='none'"
        )
        hyperparameters["latents_per_channel_weight_sharing"] = (
            [False] * len(hyperparameters['latents_per_channel'])
        )
    if (
        not isinstance(hyperparameters["latents_to_use"], list) 
        and hyperparameters["latents_to_use"] == "all"
    ):
        assert "latents_per_channel" in hyperparameters, (
            "latents_per_channel must be specified in configuration file "
            "if latents_to_use=='all'"
        )
        hyperparameters["latents_to_use"] = (
            [True] * sum(hyperparameters['latents_per_channel'])
        )
    if (
        not isinstance(hyperparameters["latents_to_optimise"], list) 
        and hyperparameters["latents_to_optimise"] == "all"
    ):
        assert "latents_per_channel" in hyperparameters, (
            "latents_per_channel must be specified in configuration file "
            "if latents_to_optimise=='all'"
        )
        hyperparameters["latents_to_optimise"] = (
            [True] * sum(hyperparameters['latents_per_channel'])
        )

        
def main():
    cli_args = parse_command_line_arguments()
    if not cli_args.json_config_file.exists():
        raise ValueError(f"No configuration file found at {cli_args.json_config_file}")
    if not cli_args.nifti_flair_dir.exists():
        raise ValueError(f"nift_flair_dir {cli_args.nifti_flair_dir} does not exist")
    if not cli_args.nifti_flair_dir.is_dir():
        raise ValueError(f"nift_flair_dir {cli_args.nifti_flair_dir} is not a directory")
    if not cli_args.output_dir.exists():
        os.makedirs(cli_args.output_dir)
    with open(cli_args.json_config_file, "r") as f:
        try:
            hyperparameters = json.load(f)
        except json.JSONDecodeError as e:
            raise json.JSONDecodeError(
                f"Configuration file {cli_args.json_config_file} not valid JSON"
            ) from e
    post_process_hyperparameters(hyperparameters, cli_args)
    torch.multiprocessing.set_start_method("spawn")
    training_script.main(hyperparameters)    


if __name__ == "__main__":
    main()