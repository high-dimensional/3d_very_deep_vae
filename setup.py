from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()
long_description = (here / 'README.md').read_text(encoding='utf-8')

install_requires = [
    'h5py==3.5.0',
    'matplotlib==3.4.3',
    'monai==0.7.0',
    'nibabel==3.2.1',
    'numpy==1.21.3',
    'Pillow==8.4.0',
    'scipy==1.7.1',
    'scikit-image==0.18.3',
    'tensorboard==2.7.0',
    'torchio==0.18.57',
    'tqdm==4.62.3',
]

# CUDA builds of torch & torchvision packages are required therefore we install directly
# from platform dependent pre-built wheels available at https://download.pytorch.org
# These are only available for 64-bit Linux and Windows systems
CUDA_VERSION = '113'  # CUDA Toolkit version to get wheels for _without_ decimal point
for os, machine in (('linux', 'x86_64'), ('win', 'AMD64')):
    for python_version in ('3.7', '3.8', '3.9'):
        base_url = f'https://download.pytorch.org/whl/cu{CUDA_VERSION}'
        abi = f'cp{python_version[0]}{python_version[-1]}'
        abi_2 = abi if python_version != '3.7' else f'{abi}m'
        local_version_label = f'cu{CUDA_VERSION}-{abi}-{abi_2}-{os}_{machine.lower()}'
        sys_platform = 'win32' if os == 'win' else os
        environment_marker = (
            f'implementation_name == "cpython" '
            f'and python_version == "{python_version}.*" '
            f'and sys_platform == "{sys_platform}" '
            f'and platform_machine == "{machine}"'
        )
        for package, package_version in (('torch', '1.10.0'), ('torchvision', '0.11.1')):
            install_requires.append(
                f'{package} @ {base_url}/{package}-{package_version}%2B'
                f'{local_version_label}.whl ; {environment_marker}'
            )

setup(
    name='verydeepvae',
    version='0.1.0',
    description='Very deep variational autoencoder models for 3D images in PyTorch',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/r-gray/3d_very_deep_vae',
    author='Robert Gray',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Image Processing',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3 :: Only',
        'Programming Language :: Python :: Implementation :: CPython',
        'Environment :: GPU :: NVIDIA CUDA :: 11.3',
        'Environment :: GPU :: NVIDIA CUDA :: 11.4',
        'Environment :: GPU :: NVIDIA CUDA :: 11.5',
        'Operating System :: POSIX :: Linux',
        'Operating System :: Microsoft :: Windows',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
    ],
    keywords='pytorch, variational autoencoder, 3d',
    license_files=('LICENSE',),
    packages=find_packages(),
    python_requires='>=3.7, <3.10',
    install_requires=install_requires,
)
