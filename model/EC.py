from model import *

encoder_params = [
    [
        OrderedDict({'conv1_leaky_1': [1, 8, 7, 2, 3]}),
        OrderedDict({'conv2_leaky_1': [64, 192, 5, 2, 2]}),
        OrderedDict({'conv3_leaky_1': [192, 192, 3, 2, 1]}),
    ],

    [
        TrajGRU(input_channel=8, num_filter=64, b_h_w=(batch_size, 128, 128), zoneout=0.0, L=13,
                i2h_kernel=(3, 3), i2h_stride=(1, 1), i2h_pad=(1, 1),
                h2h_kernel=(5, 5), h2h_dilate=(1, 1),
                act_type=activation('leaky', negative_slope=0.2, inplace=True)),

        TrajGRU(input_channel=192, num_filter=192, b_h_w=(batch_size, 64, 64), zoneout=0.0, L=13,
                i2h_kernel=(3, 3), i2h_stride=(1, 1), i2h_pad=(1, 1),
                h2h_kernel=(5, 5), h2h_dilate=(1, 1),
                act_type=activation('leaky', negative_slope=0.2, inplace=True)),
        TrajGRU(input_channel=192, num_filter=192, b_h_w=(batch_size, 32, 32), zoneout=0.0, L=9,
                i2h_kernel=(3, 3), i2h_stride=(1, 1), i2h_pad=(1, 1),
                h2h_kernel=(3, 3), h2h_dilate=(1, 1),
                act_type=activation('leaky', negative_slope=0.2, inplace=True))
    ]
]

forecaster_params = [
    [
        OrderedDict({'deconv1_leaky_1': [192, 192, 4, 2, 1]}),
        OrderedDict({'deconv2_leaky_1': [192, 64, 6, 2, 2]}),
        OrderedDict({
            'deconv3_leaky_1': [64, 32, 8, 2, 3],
            'conv3_leaky_2': [32, 16, 3, 1, 1],
            'conv3_3': [16, 8, 1, 1, 0]
        }),
    ],

    [
        TrajGRU(input_channel=192, num_filter=192, b_h_w=(batch_size, 32, 32), zoneout=0.0, L=13,
                i2h_kernel=(3, 3), i2h_stride=(1, 1), i2h_pad=(1, 1),
                h2h_kernel=(3, 3), h2h_dilate=(1, 1),
                act_type=activation('leaky', negative_slope=0.2, inplace=True)),

        TrajGRU(input_channel=192, num_filter=192, b_h_w=(batch_size, 64, 64), zoneout=0.0, L=13,
                i2h_kernel=(3, 3), i2h_stride=(1, 1), i2h_pad=(1, 1),
                h2h_kernel=(5, 5), h2h_dilate=(1, 1),
                act_type=activation('leaky', negative_slope=0.2, inplace=True)),
        TrajGRU(input_channel=64, num_filter=64, b_h_w=(batch_size, 128, 128), zoneout=0.0, L=9,
                i2h_kernel=(3, 3), i2h_stride=(1, 1), i2h_pad=(1, 1),
                h2h_kernel=(5, 5), h2h_dilate=(1, 1),
                act_type=activation('leaky', negative_slope=0.2, inplace=True))
    ]
]

class ECModel(nn.Module):

    def __init__(self):
        super(ECModel, self).__init__()
        self.encoder = Encoder(encoder_params[0], encoder_params[1]).to(cfg.device)
        self.forecaster = Forecaster(forecaster_params[0], forecaster_params[1]).to(cfg.device)

    def forward(self, input):
        state = self.encoder(input)
        output = self.forecaster(state)
        return output
