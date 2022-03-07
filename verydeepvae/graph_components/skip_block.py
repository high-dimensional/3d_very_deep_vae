import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.cuda.amp as amp
import numpy as np


class SkipBlock(nn.Module):
    """
    The residual block for very deep VAEs minus the 1x1 convs
    """

    def __init__(self, **kwargs):
        super().__init__()

        self.eps = 1e-5
        self.kwargs = kwargs
        self.channels_in = self.kwargs["channels_in"]
        self.channels_out = self.kwargs["channels_out"]
        self.hyper_params = kwargs["hyper_params"]
        self.param_count = 0
        self.hidden_kernel_size = 1

        if 'data_is_3d' in self.hyper_params and self.hyper_params['data_is_3d']:
            ConvOp = nn.Conv3d
            d = 3
            filter_space_dims = str(self.hidden_kernel_size) + "x" + str(self.hidden_kernel_size) + "x" + \
                                str(self.hidden_kernel_size)
            one_dims = '1x1x1'
        else:
            ConvOp = nn.Conv2d
            d = 2
            filter_space_dims = str(self.hidden_kernel_size) + "x" + str(self.hidden_kernel_size)
            one_dims = '1x1'

        if not self.channels_in == self.channels_out:
            self.skip_con = ConvOp(in_channels=self.channels_in, out_channels=self.channels_out, kernel_size=1,
                                   stride=1, padding=0, bias=True)
            self.param_count += self.channels_in * self.channels_out

    def forward(self, data_dictionary):
        with amp.autocast(enabled=self.kwargs["half_precision"]):

            if not self.channels_in == self.channels_out:
                data_dictionary['data'] = self.skip_con(data_dictionary['data'])

        return data_dictionary
