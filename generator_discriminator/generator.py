# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmcv.runner import load_checkpoint
from mmcv.cnn import ConvModule,build_conv_layer,kaiming_init,normal_init,xavier_init
from ..builder import GENERATOR
from torch.nn import init

def generation_init_weights(module, init_type='normal', init_gain=0.02):
    """Default initialization of network weights for image generation.

    By default, we use normal init, but xavier and kaiming might work
    better for some applications.

    Args:
        module (nn.Module): Module to be initialized.
        init_type (str): The name of an initialization method:
            normal | xavier | kaiming | orthogonal.
        init_gain (float): Scaling factor for normal, xavier and
            orthogonal.
    """

    def init_func(m):
        """Initialization function.

        Args:
            m (nn.Module): Module to be initialized.
        """
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1
                                     or classname.find('Linear') != -1):
            if init_type == 'normal':
                normal_init(m, 0.0, init_gain)
            elif init_type == 'xavier':
                xavier_init(m, gain=init_gain, distribution='normal')
            elif init_type == 'kaiming':
                kaiming_init(
                    m,
                    a=0,
                    mode='fan_in',
                    nonlinearity='leaky_relu',
                    distribution='normal')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight, gain=init_gain)
                init.constant_(m.bias.data, 0.0)
            else:
                raise NotImplementedError(
                    f"Initialization method '{init_type}' is not implemented")
        elif classname.find('BatchNorm2d') != -1:
            # BatchNorm Layer's weight is not a matrix;
            # only normal distribution applies.
            normal_init(m, 1.0, init_gain)

    module.apply(init_func)


class ResidualBlockWithDropout(nn.Module):
    """Define a Residual Block with dropout layers.

    Ref:
    Deep Residual Learning for Image Recognition

    A residual block is a conv block with skip connections. A dropout layer is
    added between two common conv modules.

    Args:
        channels (int): Number of channels in the conv layer.
        padding_mode (str): The name of padding layer:
            'reflect' | 'replicate' | 'zeros'.
        norm_cfg (dict): Config dict to build norm layer. Default:
            `dict(type='IN')`.
        use_dropout (bool): Whether to use dropout layers. Default: True.
    """

    def __init__(self,
                 channels,
                 padding_mode,
                 norm_cfg=dict(type='BN'),
                 use_dropout=True):
        super().__init__()
        assert isinstance(norm_cfg, dict), ("'norm_cfg' should be dict, but"
                                            f'got {type(norm_cfg)}')
        assert 'type' in norm_cfg, "'norm_cfg' must have key 'type'"
        # We use norm layers in the residual block with dropout layers.
        # Only for IN, use bias to follow cyclegan's original implementation.
        use_bias = norm_cfg['type'] == 'IN'

        block = [
            ConvModule(
                in_channels=channels,
                out_channels=channels,
                kernel_size=3,
                padding=1,
                bias=use_bias,
                norm_cfg=norm_cfg,
                padding_mode=padding_mode)
        ]

        if use_dropout:
            block += [nn.Dropout(0.5)]

        block += [
            ConvModule(
                in_channels=channels,
                out_channels=channels,
                kernel_size=3,
                padding=1,
                bias=use_bias,
                norm_cfg=norm_cfg,
                act_cfg=None,
                padding_mode=padding_mode)
        ]

        self.block = nn.Sequential(*block)

    def forward(self, x):
        """Forward function. Add skip connections without final ReLU.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).

        Returns:
            Tensor: Forward results.
        """
        out = x + self.block(x)
        return out



@GENERATOR.register_module()
class ResnetGenerator(nn.Module):
    """Construct a Resnet-based generator that consists of residual blocks
    between a few downsampling/upsampling operations.

    Args:
        in_channels (int): Number of channels in input images.
        out_channels (int): Number of channels in output images.
        base_channels (int): Number of filters at the last conv layer.
            Default: 64.
        norm_cfg (dict): Config dict to build norm layer. Default:
            `dict(type='IN')`.
        use_dropout (bool): Whether to use dropout layers. Default: False.
        num_blocks (int): Number of residual blocks. Default: 9.
        padding_mode (str): The name of padding layer in conv layers:
            'reflect' | 'replicate' | 'zeros'. Default: 'reflect'.
        init_cfg (dict): Config dict for initialization.
            `type`: The name of our initialization method. Default: 'normal'.
            `gain`: Scaling factor for normal, xavier and orthogonal.
            Default: 0.02.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 base_channels=64,
                 norm_cfg=dict(type='IN'),
                 use_dropout=False,
                 num_blocks=9,
                 padding_mode='reflect',
                 init_cfg=dict(type='normal', gain=0.02)):
        super().__init__()
        assert num_blocks >= 0, ('Number of residual blocks must be '
                                 f'non-negative, but got {num_blocks}.')
        assert isinstance(norm_cfg, dict), ("'norm_cfg' should be dict, but"
                                            f'got {type(norm_cfg)}')
        assert 'type' in norm_cfg, "'norm_cfg' must have key 'type'"
        # We use norm layers in the resnet generator.
        # Only for IN, use bias to follow cyclegan's original implementation.
        use_bias = norm_cfg['type'] == 'IN'

        model = []
        model += [
            ConvModule(
                in_channels=in_channels,
                out_channels=base_channels,
                kernel_size=7,
                padding=3,
                bias=use_bias,
                norm_cfg=norm_cfg,
                padding_mode=padding_mode)
        ]

        num_down = 2
        # add downsampling layers
        for i in range(num_down):
            multiple = 2**i
            model += [
                ConvModule(
                    in_channels=base_channels * multiple,
                    out_channels=base_channels * multiple * 2,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    bias=use_bias,
                    norm_cfg=norm_cfg)
            ]

        # add residual blocks
        multiple = 2**num_down
        for i in range(num_blocks):
            model += [
                ResidualBlockWithDropout(
                    base_channels * multiple,
                    padding_mode=padding_mode,
                    norm_cfg=norm_cfg,
                    use_dropout=use_dropout)
            ]

        # add upsampling layers
        for i in range(num_down):
            multiple = 2**(num_down - i)
            model += [
                ConvModule(
                    in_channels=base_channels * multiple,
                    out_channels=base_channels * multiple // 2,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    bias=use_bias,
                    conv_cfg=dict(type='deconv', output_padding=1),
                    norm_cfg=norm_cfg)
            ]

        model += [
            ConvModule(
                in_channels=base_channels,
                out_channels=out_channels,
                kernel_size=7,
                padding=3,
                bias=True,
                norm_cfg=None,
                act_cfg=dict(type='Tanh'),
                padding_mode=padding_mode)
        ]

        self.model = nn.Sequential(*model)
        self.init_type = 'normal' if init_cfg is None else init_cfg.get(
            'type', 'normal')
        self.init_gain = 0.02 if init_cfg is None else init_cfg.get(
            'gain', 0.02)

    def forward(self, x):
        """Forward function.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).

        Returns:
            Tensor: Forward results.
        """
        return self.model(x)

    def init_weights(self, pretrained=None, strict=True):
        """Initialize weights for the model.

        Args:
            pretrained (str, optional): Path for pretrained weights. If given
                None, pretrained weights will not be loaded. Default: None.
            strict (bool, optional): Whether to allow different params for the
                model and checkpoint. Default: True.
        """
        if isinstance(pretrained, str):
            pass


        elif pretrained is None:
            generation_init_weights(
                self, init_type=self.init_type, init_gain=self.init_gain)
        else:
            raise TypeError("'pretrained' must be a str or None. "
                            f'But received {type(pretrained)}.')



