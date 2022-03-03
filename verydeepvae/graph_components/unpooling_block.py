import torch.nn as nn
import torch.cuda.amp as amp
from ..misc import misc


class UnPoolingBlock(nn.Module):
    """
    Residual block for very deep VAE
    """
    def __init__(self, **kwargs):
        super().__init__()

        self.eps = 1e-5
        self.kwargs = kwargs
        self.hyper_params = self.kwargs["hyper_params"]

        if 'output_size' in self.kwargs:
            self.resampler = nn.Upsample(size=self.kwargs['output_size'], mode='nearest')
        else:
            self.resampler = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, input_dictionary):
        with amp.autocast(enabled=self.kwargs["half_precision"]):
            input_dictionary['data'] = self.resampler(input_dictionary['data'])

            if misc.key_is_true(self.hyper_params, 'separate_prior_data_stream'):
                if 'data_prior' in input_dictionary:
                    input_dictionary['data_prior'] = self.resampler(input_dictionary['data_prior'])

        return input_dictionary
