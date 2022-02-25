import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.cuda.amp as amp
import numpy as np


class SigmoidBlock(nn.Module):
    """
    """
    def __init__(self, **kwargs):
        super().__init__()
        
        self.kwargs = kwargs

    def forward(self, data_dictionary):
        with amp.autocast(enabled=self.kwargs["half_precision"]):

            data_dictionary['data'] = torch.sigmoid(data_dictionary['data'])

        return data_dictionary