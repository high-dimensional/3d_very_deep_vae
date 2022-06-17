import torch
import torch.nn as nn
import torch.cuda.amp as amp


class TanhFirstNChansOnlyBlock(nn.Module):
    """ """

    def __init__(self, **kwargs):
        super().__init__()

        self.kwargs = kwargs

        self.channels_to_tanh = self.kwargs["channels_to_tanh"]

    def forward(self, data_dictionary):
        with amp.autocast(enabled=self.kwargs["half_precision"]):
            data_dictionary["data"][:, 0 : self.channels_to_tanh, ...] = torch.tanh(
                data_dictionary["data"][:, 0 : self.channels_to_tanh, ...]
            )

        return data_dictionary
