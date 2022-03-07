import torch
import torch.nn as nn
import torch.cuda.amp as amp


class SigmoidFirstChanOnlyBlock(nn.Module):
    """
    """
    def __init__(self, **kwargs):
        super().__init__()
        
        self.kwargs = kwargs

    def forward(self, data_dictionary):
        with amp.autocast(enabled=self.kwargs["half_precision"]):
            data_dictionary['data'][:, 0, ...] = torch.sigmoid(data_dictionary['data'][:, 0, ...])

        return data_dictionary