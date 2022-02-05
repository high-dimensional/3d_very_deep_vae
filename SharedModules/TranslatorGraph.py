import torch.nn as nn
from NotNeeded.cnn_encoder import CnnEncoder
from NotNeeded.cnn_decoder import CnnDecoder


class Graph:
    """
    
    """
    def __init__(self, **kwargs):
        super().__init__()

        # b0 -> [T1; FLAIR]
        self.encoder_conv_1 = CnnEncoder(standardise_input_using_ema=False,
                                         is_colour=False,
                                         is_3d=kwargs["hyper_params"]['data_is_3d'],
                                         output_channels=kwargs["hyper_params"]['channels'],
                                         kernel_shapes=[3] * len(kwargs["hyper_params"]['channels']),
                                         pooling_per_layer=kwargs["hyper_params"]['pooling_per_layer'],
                                         hidden_act=nn.LeakyReLU(negative_slope=1e-2),
                                         # hidden_act=nn.GELU(),
                                         batch_norm=kwargs["hyper_params"]['batch_norm_conv_layers'],
                                         batch_norm_momentum=kwargs["hyper_params"]['batch_norm_momentum'],
                                         nonparametric_resampling=kwargs["hyper_params"]['nonparametric_resampling'],
                                         half_precision=kwargs["hyper_params"]['half_precision'],
                                         hyper_params=kwargs["hyper_params"])
        self.decoder_conv_1 = CnnDecoder(override_value_for_channel_0=2,
                                         is_colour=False,
                                         is_3d=kwargs["hyper_params"]['data_is_3d'],
                                         output_channels=kwargs["hyper_params"]['channels'],
                                         kernel_shapes=[3] * len(kwargs["hyper_params"]['channels']),
                                         pooling_per_layer=kwargs["hyper_params"]['pooling_per_layer'],
                                         hidden_act=nn.LeakyReLU(negative_slope=1e-2),
                                         # hidden_act=nn.GELU(),
                                         output_act=nn.Identity(),
                                         batch_norm=kwargs["hyper_params"]['batch_norm_conv_layers'],
                                         batch_norm_momentum=kwargs["hyper_params"]['batch_norm_momentum'],
                                         nonparametric_resampling=kwargs["hyper_params"]['nonparametric_resampling'],
                                         half_precision=kwargs["hyper_params"]['half_precision'])

        # T1 -> b1000
        self.encoder_conv_2 = CnnEncoder(standardise_input_using_ema=False,
                                         is_colour=False,
                                         is_3d=kwargs["hyper_params"]['data_is_3d'],
                                         output_channels=kwargs["hyper_params"]['channels'],
                                         kernel_shapes=[3] * len(kwargs["hyper_params"]['channels']),
                                         pooling_per_layer=kwargs["hyper_params"]['pooling_per_layer'],
                                         hidden_act=nn.LeakyReLU(negative_slope=1e-2),
                                         # hidden_act=nn.GELU(),
                                         batch_norm=kwargs["hyper_params"]['batch_norm_conv_layers'],
                                         batch_norm_momentum=kwargs["hyper_params"]['batch_norm_momentum'],
                                         nonparametric_resampling=kwargs["hyper_params"]['nonparametric_resampling'],
                                         half_precision=kwargs["hyper_params"]['half_precision'],
                                         hyper_params=kwargs["hyper_params"])
        self.decoder_conv_2 = CnnDecoder(is_colour=False,
                                         is_3d=kwargs["hyper_params"]['data_is_3d'],
                                         output_channels=kwargs["hyper_params"]['channels'],
                                         kernel_shapes=[3] * len(kwargs["hyper_params"]['channels']),
                                         pooling_per_layer=kwargs["hyper_params"]['pooling_per_layer'],
                                         hidden_act=nn.LeakyReLU(negative_slope=1e-2),
                                         # hidden_act=nn.GELU(),
                                         output_act=nn.Identity(),
                                         batch_norm=kwargs["hyper_params"]['batch_norm_conv_layers'],
                                         batch_norm_momentum=kwargs["hyper_params"]['batch_norm_momentum'],
                                         nonparametric_resampling=kwargs["hyper_params"]['nonparametric_resampling'],
                                         half_precision=kwargs["hyper_params"]['half_precision'])

        # b0 -> b1000
        self.encoder_conv_3 = CnnEncoder(standardise_input_using_ema=False,
                                         is_colour=False,
                                         is_3d=kwargs["hyper_params"]['data_is_3d'],
                                         output_channels=kwargs["hyper_params"]['channels'],
                                         kernel_shapes=[3] * len(kwargs["hyper_params"]['channels']),
                                         pooling_per_layer=kwargs["hyper_params"]['pooling_per_layer'],
                                         hidden_act=nn.LeakyReLU(negative_slope=1e-2),
                                         # hidden_act=nn.GELU(),
                                         batch_norm=kwargs["hyper_params"]['batch_norm_conv_layers'],
                                         batch_norm_momentum=kwargs["hyper_params"]['batch_norm_momentum'],
                                         nonparametric_resampling=kwargs["hyper_params"]['nonparametric_resampling'],
                                         half_precision=kwargs["hyper_params"]['half_precision'],
                                         hyper_params=kwargs["hyper_params"])
        self.decoder_conv_3 = CnnDecoder(is_colour=False,
                                         is_3d=kwargs["hyper_params"]['data_is_3d'],
                                         output_channels=kwargs["hyper_params"]['channels'],
                                         kernel_shapes=[3] * len(kwargs["hyper_params"]['channels']),
                                         pooling_per_layer=kwargs["hyper_params"]['pooling_per_layer'],
                                         hidden_act=nn.LeakyReLU(negative_slope=1e-2),
                                         # hidden_act=nn.GELU(),
                                         output_act=nn.Identity(),
                                         batch_norm=kwargs["hyper_params"]['batch_norm_conv_layers'],
                                         batch_norm_momentum=kwargs["hyper_params"]['batch_norm_momentum'],
                                         nonparametric_resampling=kwargs["hyper_params"]['nonparametric_resampling'],
                                         half_precision=kwargs["hyper_params"]['half_precision'])

        # b0 -> T1
        self.encoder_conv_4 = CnnEncoder(standardise_input_using_ema=False,
                                         is_colour=False,
                                         is_3d=kwargs["hyper_params"]['data_is_3d'],
                                         output_channels=kwargs["hyper_params"]['channels'],
                                         kernel_shapes=[3] * len(kwargs["hyper_params"]['channels']),
                                         pooling_per_layer=kwargs["hyper_params"]['pooling_per_layer'],
                                         hidden_act=nn.LeakyReLU(negative_slope=1e-2),
                                         # hidden_act=nn.GELU(),
                                         batch_norm=kwargs["hyper_params"]['batch_norm_conv_layers'],
                                         batch_norm_momentum=kwargs["hyper_params"]['batch_norm_momentum'],
                                         nonparametric_resampling=kwargs["hyper_params"]['nonparametric_resampling'],
                                         half_precision=kwargs["hyper_params"]['half_precision'],
                                         hyper_params=kwargs["hyper_params"])
        self.decoder_conv_4 = CnnDecoder(is_colour=False,
                                         is_3d=kwargs["hyper_params"]['data_is_3d'],
                                         output_channels=kwargs["hyper_params"]['channels'],
                                         kernel_shapes=[3] * len(kwargs["hyper_params"]['channels']),
                                         pooling_per_layer=kwargs["hyper_params"]['pooling_per_layer'],
                                         hidden_act=nn.LeakyReLU(negative_slope=1e-2),
                                         # hidden_act=nn.GELU(),
                                         output_act=nn.Identity(),
                                         batch_norm=kwargs["hyper_params"]['batch_norm_conv_layers'],
                                         batch_norm_momentum=kwargs["hyper_params"]['batch_norm_momentum'],
                                         nonparametric_resampling=kwargs["hyper_params"]['nonparametric_resampling'],
                                         half_precision=kwargs["hyper_params"]['half_precision'])

        # [T1; FLAIR] -> b1000
        self.encoder_conv_6 = CnnEncoder(override_value_for_channel_0=2,
                                         standardise_input_using_ema=False,
                                         is_colour=False,
                                         is_3d=kwargs["hyper_params"]['data_is_3d'],
                                         output_channels=kwargs["hyper_params"]['channels'],
                                         kernel_shapes=[3] * len(kwargs["hyper_params"]['channels']),
                                         pooling_per_layer=kwargs["hyper_params"]['pooling_per_layer'],
                                         hidden_act=nn.LeakyReLU(negative_slope=1e-2),
                                         # hidden_act=nn.GELU(),
                                         batch_norm=kwargs["hyper_params"]['batch_norm_conv_layers'],
                                         batch_norm_momentum=kwargs["hyper_params"]['batch_norm_momentum'],
                                         nonparametric_resampling=kwargs["hyper_params"]['nonparametric_resampling'],
                                         half_precision=kwargs["hyper_params"]['half_precision'],
                                         hyper_params=kwargs["hyper_params"])
        self.decoder_conv_6 = CnnDecoder(is_colour=False,
                                         is_3d=kwargs["hyper_params"]['data_is_3d'],
                                         output_channels=kwargs["hyper_params"]['channels'],
                                         kernel_shapes=[3] * len(kwargs["hyper_params"]['channels']),
                                         pooling_per_layer=kwargs["hyper_params"]['pooling_per_layer'],
                                         hidden_act=nn.LeakyReLU(negative_slope=1e-2),
                                         # hidden_act=nn.GELU(),
                                         output_act=nn.Identity(),
                                         batch_norm=kwargs["hyper_params"]['batch_norm_conv_layers'],
                                         batch_norm_momentum=kwargs["hyper_params"]['batch_norm_momentum'],
                                         nonparametric_resampling=kwargs["hyper_params"]['nonparametric_resampling'],
                                         half_precision=kwargs["hyper_params"]['half_precision'])

        self.encoder_1 = nn.Sequential(self.encoder_conv_1).to(kwargs["device"])
        self.decoder_1 = nn.Sequential(self.decoder_conv_1).to(kwargs["device"])

        self.encoder_2 = nn.Sequential(self.encoder_conv_2).to(kwargs["device"])
        self.decoder_2 = nn.Sequential(self.decoder_conv_2).to(kwargs["device"])
        
        self.encoder_3 = nn.Sequential(self.encoder_conv_3).to(kwargs["device"])
        self.decoder_3 = nn.Sequential(self.decoder_conv_3).to(kwargs["device"])

        self.encoder_4 = nn.Sequential(self.encoder_conv_4).to(kwargs["device"])
        self.decoder_4 = nn.Sequential(self.decoder_conv_4).to(kwargs["device"])

        self.encoder_6 = nn.Sequential(self.encoder_conv_6).to(kwargs["device"])
        self.decoder_6 = nn.Sequential(self.decoder_conv_6).to(kwargs["device"])
