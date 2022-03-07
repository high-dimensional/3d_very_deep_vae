import torch
import numpy as np
import torch.distributed as dist
import os
from ..misc import misc
from ..data_tools import data_handling
import socket
import monai
import torchio
import torchvision


def setup_environment(hyper_params):
    """
    See https://pytorch.org/docs/stable/elastic/run.html for a list of environmental variables created by
    torch.distributed.run
    """
    hyper_params['host_name'] = socket.gethostname()
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(hyper_params['CUDA_devices'])

    if 'LOCAL_WORLD_SIZE' in os.environ and os.environ['LOCAL_WORLD_SIZE'] is not None:
        """
        The presence of these environmental variables suggests we have a dist.run startup
        """
        local_rank = int(os.environ['LOCAL_RANK'])
        hyper_params['local_rank'] = local_rank
        misc.print_0(hyper_params, "Found local_rank in LOCAL_RANK env var: " + str(local_rank))

        global_rank = int(os.environ['RANK'])
        hyper_params['global_rank'] = global_rank
        misc.print_0(hyper_params, "Found global_rank in RANK env var: " + str(global_rank))

        global_world_size = int(os.environ['WORLD_SIZE'])
        hyper_params['global_world_size'] = global_world_size
        misc.print_0(hyper_params, "Found global_world_size in WORLD_SIZE env var: " + str(global_world_size))

        world_size = int(os.environ['LOCAL_WORLD_SIZE'])
        hyper_params['world_size'] = world_size
        misc.print_0(hyper_params, "Found world_size in LOCAL_WORLD_SIZE env var: " + str(world_size))

        master_addr = str(os.environ['MASTER_ADDR'])
        hyper_params['master_addr'] = master_addr
        misc.print_0(hyper_params, "Found master_addr in MASTER_ADDR env var: " + str(master_addr))

        master_port = int(os.environ['MASTER_PORT'])
        hyper_params['master_port'] = master_port
        misc.print_0(hyper_params, "Found master_port in MASTER_ADDR env var: " + str(master_port))
    else:
        if hasattr(hyper_params['args'], 'local_rank'):
            local_rank = hyper_params['args'].local_rank
            hyper_params['local_rank'] = local_rank
            if local_rank is None:
                print("local_rank is NONE: quitting")
                quit()
            else:
                misc.print_0(hyper_params, "Found local_rank in input args: " + str(local_rank))
        else:
            local_rank = hyper_params['local_rank']
            misc.print_0(hyper_params, "Found local_rank in hyper_params: " + str(local_rank))

        world_size = hyper_params['world_size']

        if 'master_addr' in hyper_params:
            master_addr = hyper_params['master_addr']
        else:
            master_addr = '127.0.0.1'
            hyper_params['master_addr'] = master_addr

        if 'master_port' in hyper_params:
            master_port = hyper_params['master_port']
        else:
            master_port = 1234
            hyper_params['master_port'] = master_port

        os.environ['MASTER_ADDR'] = master_addr
        os.environ['MASTER_PORT'] = str(master_port)

        misc.print_0(hyper_params, "Found world_size in hyper_params: " + str(world_size))
        misc.print_0(hyper_params, "master_addr: " + str(master_addr))
        misc.print_0(hyper_params, "master_port: " + str(master_port))

        hyper_params['global_world_size'] = hyper_params['world_size']
        hyper_params['global_rank'] = hyper_params['local_rank']

    os.environ['OMP_NUM_THREADS'] = str(hyper_params['workers_per_process'])

    device = torch.device('cuda', local_rank)
    hyper_params['device'] = device
    device_ids = [local_rank]
    hyper_params['device_ids'] = device_ids
    torch.cuda.set_device(device)

    if local_rank > 0:
        hyper_params['verbose'] = False

    if 'threads_per_rank' in hyper_params:
        torch.set_num_threads(hyper_params['threads_per_rank'])
        misc.print_0(hyper_params, f"Setting number of threads to {hyper_params['threads_per_rank']}")
    else:
        misc.print_0(hyper_params, "Setting number of threads to 8")
        torch.set_num_threads(8)

    misc.print_0(hyper_params, "NB: Only output from the rank_0 process is displayed")
    misc.print_0(hyper_params, "Master IP: " + str(hyper_params['master_addr']))
    misc.print_0(hyper_params, "Master Port: " + str(hyper_params['master_port']))
    misc.print_0(hyper_params, "Workers per process: " + str(hyper_params['workers_per_process']))

    if world_size > 1:
        misc.print_0(hyper_params, "Waiting for all DDP processes to establish contact...", end="")

    # from datetime import timedelta
    # if 'LOCAL_WORLD_SIZE' in os.environ:
    #     dist.init_process_group(backend='nccl', timeout=timedelta(seconds=3))
    # else:
    #     dist.init_process_group(backend='nccl', timeout=timedelta(seconds=3), rank=local_rank,
    #                                          world_size=world_size)

    if dist.is_nccl_available():
        backend = 'nccl'
    else:
        backend = 'gloo'

    # from datetime import timedelta
    if 'LOCAL_WORLD_SIZE' in os.environ:
        dist.init_process_group(backend=backend)
    else:
        dist.init_process_group(backend=backend, rank=local_rank, world_size=world_size)

    if world_size > 1:
        misc.print_0(hyper_params, "contact established!")

    misc.print_0(hyper_params, "Multiprocessing start method: " + torch.multiprocessing.get_start_method())

    misc.print_0(hyper_params, "Torch version: " + torch.__version__)
    misc.print_0(hyper_params, "Torchio version: " + torchio.__version__)
    misc.print_0(hyper_params, "Numpy version: " + np.__version__)
    misc.print_0(hyper_params, "Monai version: " + monai.__version__)
    misc.print_0(hyper_params, "Torchvision version: " + torchvision.__version__)
    misc.print_0(hyper_params, "Host name: " + hyper_params['host_name'])
    misc.print_0(hyper_params, "CUDA version: " + torch.version.cuda)
    misc.print_0(hyper_params, "CUDA_VISIBLE_DEVICES: " + os.environ['CUDA_VISIBLE_DEVICES'])
    misc.print_0(hyper_params, "Backend: " + backend)

    hyper_params['recon_folder'] = os.path.join(hyper_params['current_dir'], 'output')
    hyper_params['samples_folder'] = os.path.join(hyper_params['recon_folder'], 'samples')

    # Checking folder exist is handled by the rank_0 process only!
    if local_rank == 0:
        if not os.path.exists(hyper_params['tensorboard_dir']):
            os.makedirs(hyper_params['tensorboard_dir'])
        if not os.path.exists(hyper_params['recon_folder']):
            os.makedirs(hyper_params['recon_folder'])
        if not os.path.exists(hyper_params['samples_folder']):
            os.makedirs(hyper_params['samples_folder'])
        if not os.path.exists(hyper_params['checkpoint_folder']):
            os.makedirs(hyper_params['checkpoint_folder'])

        misc.print_0(hyper_params,
                     "To start TB: tensorboard --logdir " + hyper_params['tensorboard_dir'] + " --port=8008")
        if not (hyper_params['resume_from_checkpoint'] and hyper_params['restore_optimiser']):
            data_handling.delete_directory_contents(hyper_params['tensorboard_dir'])

    misc.print_0(hyper_params, "Checkpoint folder: " + hyper_params['checkpoint_folder'])

    return hyper_params
