import torch.nn as nn
import torch.cuda.amp as amp


class ResBlock(nn.Module):
    """
    Residual block for very deep VAEs
    """

    def __init__(self, **kwargs):
        super().__init__()

        self.eps = 1e-5
        self.kwargs = kwargs
        self.lateral_skip_con_index = None
        self.depth = self.kwargs["depth"]
        self.channels_in = self.kwargs["channels_in"]
        self.channels_hidden = self.kwargs["channels_hidden"]
        self.channels_out = self.kwargs["channels_out"]

        if (
            "veto_skip_connection" in self.kwargs
            and self.kwargs["veto_skip_connection"]
        ):
            self.veto_skip_connection = True
        else:
            self.veto_skip_connection = False

        self.hidden_activation = nn.GELU()
        self.visible_activation = None
        if "activation" in self.kwargs:
            self.visible_activation = self.kwargs["activation"]

        if "lateral_skip_con_index" in self.kwargs:
            self.lateral_skip_con_index = self.kwargs["lateral_skip_con_index"]

        self.pad = nn.ConstantPad2d((1, 1, 1, 1), 0)
        self.convolutions = nn.ModuleList()

        self.hidden_kernel_size = 3
        if "hidden_kernel_size" in self.kwargs:
            self.hidden_kernel_size = self.kwargs["hidden_kernel_size"]

        self.convolutions.append(
            nn.Conv2d(
                in_channels=self.channels_in,
                out_channels=self.channels_hidden,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True,
            )
        )
        self.convolutions.append(
            nn.Conv2d(
                in_channels=self.channels_hidden,
                out_channels=self.channels_hidden,
                kernel_size=self.hidden_kernel_size,
                stride=1,
                padding=0,
                bias=True,
            )
        )
        self.convolutions.append(
            nn.Conv2d(
                in_channels=self.channels_hidden,
                out_channels=self.channels_hidden,
                kernel_size=self.hidden_kernel_size,
                stride=1,
                padding=0,
                bias=True,
            )
        )
        self.convolutions.append(
            nn.Conv2d(
                in_channels=self.channels_hidden,
                out_channels=self.channels_out,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True,
            )
        )

        # self.batch_norm = nn.SyncBatchNorm(num_features=self.channels_in, eps=1e-5, momentum=0.1, affine=True,
        #                                    track_running_stats=True)
        self.batch_norm = nn.BatchNorm2d(
            num_features=self.channels_in,
            eps=1e-5,
            momentum=0.1,
            affine=True,
            track_running_stats=True,
        )

        if self.channels_in == self.channels_out:
            self.skip_con = nn.Identity()
        else:
            self.skip_con = nn.Conv2d(
                in_channels=self.channels_in,
                out_channels=self.channels_out,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True,
            )

    def forward(self, data_dictionary):
        with amp.autocast(enabled=self.kwargs["half_precision"]):

            copy_of_incoming_data = data_dictionary["data"].clone()

            data = data_dictionary["data"]
            data = self.batch_norm(data)
            data = self.hidden_activation(data)

            data = self.convolutions[0](data)
            if self.hidden_kernel_size == 3:
                data = self.convolutions[1](self.pad(data))
                data = self.convolutions[2](self.pad(data))
            else:
                data = self.convolutions[1](data)
                data = self.convolutions[2](data)
            data = self.convolutions[3](data)

            if self.visible_activation is not None:
                # The .clone() stops the error '...SigmoidBackward, is at version 1;
                # expected version 0 instead' when using a nn.Sigmoid() activation.
                data = self.visible_activation(data).clone()

            if not self.veto_skip_connection:
                # For the output layers we veto the skip connection to ensure the image of the block is the image of the
                # ResBlock's activation
                data += self.skip_con(copy_of_incoming_data)

            data_dictionary["data"] = data

            if self.lateral_skip_con_index is not None:
                data_dictionary[
                    "lateral_skip_" + str(self.lateral_skip_con_index)
                ] = data

        return data_dictionary
