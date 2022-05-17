"""Generate NIfTI images as synthetic training data set."""

import argparse
import os
import pathlib
import numpy as np
import nibabel as nb
import tqdm
from scipy.ndimage import gaussian_filter

def generate_synthetic_voxels(rng, args):
    shape = (args.voxels_per_axis,) * 3
    voxels = gaussian_filter(
        rng.uniform(0, args.background_noise_amplitude, size=shape),
        sigma=args.background_noise_length_scale,
    )
    principle_axes_half_lengths = rng.integers(
        low=max(1, args.voxels_per_axis // 8),
        high=max(1, args.voxels_per_axis // 2),
        size=3,
        endpoint=True,
    )
    random_rotation_matrix, _ = np.linalg.qr(rng.standard_normal((3, 3)))
    voxel_coords = np.stack(
        np.meshgrid(*((np.arange(args.voxels_per_axis),) * 3), indexing="ij"), axis=-1
    ) - np.array(((args.voxels_per_axis - 1) / 2,) * 3)
    rotated_voxel_coords = voxel_coords @ random_rotation_matrix
    ellipse_mask = (
        np.sum(rotated_voxel_coords ** 2 / principle_axes_half_lengths ** 2, -1) < 1
    )
    voxels[ellipse_mask] += rng.uniform(*args.ellipsoid_increment_interval)
    voxels /= voxels.max()
    assert np.all((voxels >= 0) & (voxels <= 1))
    return voxels


def main():
    parser = argparse.ArgumentParser(
        "Generate random ellipsoid inclusion in background noise NIfTI volumetric "
        "images as synthetic training data set",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--voxels_per_axis",
        type=int,
        default=32,
        help="The number of voxels along each of the three spatial axes",
    )
    parser.add_argument(
        "--number_of_files",
        type=int,
        default=1000,
        help="The number of NIfTI files to generate",
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=None,
        help="Integer seed to use to initialise pseudo-random number generator state",
    )
    parser.add_argument(
        "--output_directory",
        type=pathlib.Path,
        required=True,
        help="The directory to write the generated files to",
    )
    parser.add_argument(
        "--background_noise_amplitude",
        type=float,
        default=0.05,
        help="The amplitude in [0, 1] of the (pre-filtering) background noise",
    )
    parser.add_argument(
        "--background_noise_length_scale",
        type=float,
        default=0.5,
        help="Length-scale parameter (in voxels) of Gaussian filter applied to noise",
    )
    parser.add_argument(
        "--ellipsoid_increment_interval",
        type=float,
        nargs=2,
        default=[0.25, 0.75],
        help="The interval the value added to ellipsoid voxels is uniformly drawn from",
    )
    args = parser.parse_args()
    affine_transformation = np.identity(4)
    if not args.output_directory.exists():
        os.makedirs(args.output_directory)
    if args.number_of_files <= 0:
        raise ValueError("number_of_files must be a positive integer")
    if args.voxels_per_axis <= 0 or (np.log2(args.voxels_per_axis) % 1) != 0.0:
        raise ValueError("voxels_per_image must be a positive power of two")
    rng = np.random.default_rng(args.random_seed)
    for file_index in tqdm.trange(args.number_of_files):
        voxels = generate_synthetic_voxels(rng, args)
        nifti_image = nb.Nifti1Image(voxels, affine_transformation)
        padded_index = str(file_index).zfill(len(str(args.number_of_files - 1)))
        nb.save(nifti_image, args.output_directory / f"ellipsoid_{padded_index}.nii")


if __name__ == "__main__":
    main()
