import torch
import torch.nn as nn
import torch.cuda.amp as amp


class SigmoidFirstNChansOnlyBlock(nn.Module):
    """
    """

    def __init__(self, **kwargs):
        super().__init__()

        self.kwargs = kwargs

        self.channels_to_sigmoid = self.kwargs["channels_to_sigmoid"]

    def forward(self, data_dictionary):
        with amp.autocast(enabled=self.kwargs["half_precision"]):
            data_dictionary["data"][
                :, 0 : self.channels_to_sigmoid, ...
            ] = torch.sigmoid(
                data_dictionary["data"][:, 0 : self.channels_to_sigmoid, ...]
            )

        return data_dictionary
