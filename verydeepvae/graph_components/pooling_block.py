import torch.nn as nn
import torch.cuda.amp as amp


class PoolingBlock(nn.Module):
    """
    Pooling block for very deep VAE
    """

    def __init__(self, **kwargs):
        super().__init__()

        self.eps = 1e-5
        self.kwargs = kwargs
        self.hyper_params = self.kwargs["hyper_params"]

        if "data_is_3d" in self.hyper_params and self.hyper_params["data_is_3d"]:
            ConvOp = nn.Conv3d
        else:
            ConvOp = nn.Conv2d

        if "channels" in self.kwargs:
            self.channels = self.kwargs["channels"]
            self.resampler = ConvOp(
                in_channels=self.channels,
                out_channels=self.channels,
                kernel_size=2,
                stride=2,
                padding=0,
                bias=True,
            )
        else:
            self.resampler = nn.Upsample(scale_factor=0.5, mode="nearest")

    def forward(self, input_dictionary):
        with amp.autocast(enabled=self.kwargs["half_precision"]):

            if "channels" in self.kwargs:
                input_dictionary["data"] = self.resampler(input_dictionary["data"])
            else:
                input_dictionary["data"] = nn.functional.interpolate(
                    input_dictionary["data"],
                    scale_factor=0.5,
                    recompute_scale_factor=False,
                )

        return input_dictionary
