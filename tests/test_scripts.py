"""Run integration tests with data generation and training scripts"""

import json
import subprocess
import sys
from pathlib import Path

import pytest


SEED = 7856391


def run_script(script_filename, script_args, check_return_code=False):
    script_path = Path(__file__).parent.parent / "scripts" / script_filename
    assert script_path.exists(), f"Cannot find script at {script_path}"
    command_args = [sys.executable, str(script_path.resolve())] + script_args
    completed_process = subprocess.run(
        command_args, encoding="utf8", capture_output=True
    )
    if check_return_code:
        assert completed_process.returncode == 0, (
            f"Running {' '.join(command_args)} fails to successfully complete "
            f"with stderr output\n\n{completed_process.stderr}"
        )
    return completed_process


@pytest.mark.parametrize("voxels_per_axis", [8, 16, 32])
@pytest.mark.parametrize("number_of_files", [10, 100])
def test_generate_synthetic_data(tmp_path, voxels_per_axis, number_of_files):
    script_args = [
        "--voxels_per_axis",
        str(voxels_per_axis),
        "--number_of_files",
        str(number_of_files),
        "--output_directory",
        str(tmp_path),
        "--random_seed",
        str(SEED),
    ]
    run_script("generate_synthetic_data.py", script_args, check_return_code=True)
    assert len(list(tmp_path.glob("ellipsoid_*.nii"))) == number_of_files


@pytest.mark.parametrize("number_of_files", [-1, 0])
def test_generate_synthetic_data_non_valid_number_of_files_fails(
    tmp_path, number_of_files
):
    script_args = [
        "--voxels_per_axis",
        "8",
        "--number_of_files",
        str(number_of_files),
        "--output_directory",
        str(tmp_path),
        "--random_seed",
        str(SEED),
    ]
    completed_process = run_script("generate_synthetic_data.py", script_args)
    assert completed_process.returncode != 0
    assert "number_of_files must be a positive integer" in str(completed_process.stderr)


@pytest.mark.parametrize("voxels_per_axis", [-1, 0, 7])
def test_generate_synthetic_data_non_valid_resolution_fails(tmp_path, voxels_per_axis):
    script_args = [
        "--voxels_per_axis",
        str(voxels_per_axis),
        "--number_of_files",
        "10",
        "--output_directory",
        str(tmp_path),
        "--random_seed",
        str(SEED),
    ]
    completed_process = run_script("generate_synthetic_data.py", script_args)
    assert completed_process.returncode != 0
    assert "voxels_per_axis must be a positive power of two" in str(
        completed_process.stderr
    )


@pytest.fixture
def configuration_dict():
    return {
        "random_seed": SEED,
        "total_epochs": 1,
        "batch_size": 1,
        "resolution": 32,
        "max_niis_to_use": 2,
        "latent_feature_maps_per_resolution": [2, 4, 4, 3, 2, 1],
        "channels_per_latent": [10, 10, 10, 10, 10, 50],
        "channels": [10, 20, 30, 40, 50, 60],
        "kernel_sizes_bottom_up": [3, 3, 3, 3, 2, 1],
        "kernel_sizes_top_down": [3, 3, 3, 3, 2, 1],
        "channels_hidden": [10, 20, 30, 40, 50, 60],
        "channels_top_down": [10, 20, 30, 40, 50, 60],
        "channels_hidden_top_down": [10, 20, 30, 40, 50, 60],
    }


@pytest.fixture(scope="module")
def synthetic_data_dir(tmp_path_factory):
    voxels_per_axis = 32
    number_of_files = 2
    data_dir = tmp_path_factory.mktemp("data")
    generate_data_args = [
        "--voxels_per_axis",
        str(voxels_per_axis),
        "--number_of_files",
        str(number_of_files),
        "--output_directory",
        str(data_dir),
        "--random_seed",
        str(SEED),
    ]
    run_script("generate_synthetic_data.py", generate_data_args, check_return_code=True)
    return data_dir


def train_vae_model_with_configuration(configuration_dict, data_dir, tmp_path_factory):
    configuration_path = tmp_path_factory.mktemp("configurations") / "test_config.json"
    with open(configuration_path, "w") as f:
        json.dump(configuration_dict, f)
    outputs_dir = tmp_path_factory.mktemp("outputs")
    train_model_args = [
        "--json_config_file",
        str(configuration_path),
        "--nifti_dir",
        str(data_dir),
        "--output_dir",
        str(outputs_dir),
    ]
    run_script("train_vae_model.py", train_model_args, check_return_code=True)


@pytest.mark.parametrize(
    "configuration_updates",
    [
        {},
        {"total_epochs": 2},
        {"half_precision": True},
        {"convolutional_downsampling": True},
        {"bottleneck_resnet_encoder": False},
        {"only_use_one_conv_block_at_top": True},
        {"normalise_weight_by_depth": False},
        {"zero_biases": False},
        {"use_rezero": True},
        {"veto_transformations": True},
        {"apply_augmentations_to_validation_set": True},
        {"predict_x_scale": False},
        {"predict_x_scale_with_sigmoid": False},
        {"use_precision_reweighting": True},
        {"separate_hidden_loc_scale_convs": True},
        {"verbose": False},
        {"output_activation_function": "sigmoid"},
        {"latent_feature_maps_per_resolution_weight_sharing": "all"},
        {"latents_to_optimise": "none"},
    ],
    ids=lambda d: ",".join(f"{k}={v}" for k, v in d.items()) if d else "base",
)
def test_generate_synthetic_data_and_train_vae_model(
    tmp_path_factory, synthetic_data_dir, configuration_dict, configuration_updates
):
    configuration_dict.update(configuration_updates)
    train_vae_model_with_configuration(
        configuration_dict, synthetic_data_dir, tmp_path_factory
    )


@pytest.mark.parametrize(
    "invalid_configuration_updates",
    [
        {"total_epochs": -1},
        {"batch_size": "100.2"},
        {"foo": True},
    ],
    ids=lambda d: ",".join(f"{k}={v}" for k, v in d.items()) if d else "base",
)
def test_generate_synthetic_data_and_train_vae_model_with_invalid_config_file_raises(
    tmp_path_factory,
    synthetic_data_dir,
    configuration_dict,
    invalid_configuration_updates,
):
    configuration_dict.update(invalid_configuration_updates)
    with pytest.raises(AssertionError, match="jsonschema.exceptions.ValidationError"):
        train_vae_model_with_configuration(
            configuration_dict, synthetic_data_dir, tmp_path_factory
        )
