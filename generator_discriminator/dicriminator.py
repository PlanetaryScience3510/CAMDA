import torch.nn as nn
from ..builder import BACKBONES
from mmcv.cnn import ConvModule,build_conv_layer,kaiming_init,normal_init,xavier_init
from ..builder import Discriminator
from torch.nn import init
from mmdet.models.builder import build_dicriminator,build_loss
import torch
from collections import OrderedDict
import torch.distributed as dist
from mmdet.models.utils.grl_layer import GrlLayer
from mmcv.runner import BaseModule
import torch.nn.functional as F


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


@Discriminator.register_module()
class PatchDiscriminator(nn.Module):
    """A PatchGAN discriminator.

    Args:
        in_channels (int): Number of channels in input images.
        base_channels (int): Number of channels at the first conv layer.
            Default: 64.
        num_conv (int): Number of stacked intermediate convs (excluding input
            and output conv). Default: 3.
        norm_cfg (dict): Config dict to build norm layer. Default:
            `dict(type='BN')`.
        init_cfg (dict): Config dict for initialization.
            `type`: The name of our initialization method. Default: 'normal'.
            `gain`: Scaling factor for normal, xavier and orthogonal.
            Default: 0.02.
    """

    def __init__(self,
                 in_channels,
                 base_channels=64,
                 num_conv=3,
                 norm_cfg=dict(type='BN'),
                 init_cfg=dict(type='normal', gain=0.02)):
        super().__init__()
        assert isinstance(norm_cfg, dict), ("'norm_cfg' should be dict, but"
                                            f'got {type(norm_cfg)}')
        assert 'type' in norm_cfg, "'norm_cfg' must have key 'type'"
        # We use norm layers in the patch discriminator.
        # Only for IN, use bias since it does not have affine parameters.
        use_bias = norm_cfg['type'] == 'IN'

        kernel_size = 3
        padding = 1

        # input layer
        sequence = [
            ConvModule(
                in_channels=in_channels,
                out_channels=base_channels,
                kernel_size=kernel_size,
                stride=2,
                padding=padding,
                bias=True,
                norm_cfg=None,
                act_cfg=dict(type='LeakyReLU', negative_slope=0.2))
        ]

        # stacked intermediate layers,
        # gradually increasing the number of filters
        multiple_now = 1
        multiple_prev = 1
        for n in range(1, num_conv):
            multiple_prev = multiple_now
            multiple_now = min(2**n, 8)
            sequence += [
                ConvModule(
                    in_channels=base_channels * multiple_prev,
                    out_channels=base_channels * multiple_now,
                    kernel_size=kernel_size,
                    stride=2,
                    padding=padding,
                    bias=use_bias,
                    norm_cfg=norm_cfg,
                    act_cfg=dict(type='LeakyReLU', negative_slope=0.2))
            ]
        multiple_prev = multiple_now
        multiple_now = min(2**num_conv, 8)
        sequence += [
            ConvModule(
                in_channels=base_channels * multiple_prev,
                out_channels=base_channels * multiple_now,
                kernel_size=kernel_size,
                stride=1,
                padding=padding,
                bias=use_bias,
                norm_cfg=norm_cfg,
                act_cfg=dict(type='LeakyReLU', negative_slope=0.2))
        ]

        # output one-channel prediction map
        sequence += [
            build_conv_layer(
                dict(type='Conv2d'),
                base_channels * multiple_now,
                1,
                kernel_size=kernel_size,
                stride=1,
                padding=padding)
        ]

        self.model = nn.Sequential(*sequence)
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

    def init_weights(self, pretrained=None):
        """Initialize weights for the model.

        Args:
            pretrained (str, optional): Path for pretrained weights. If given
                None, pretrained weights will not be loaded. Default: None.
        """
        if isinstance(pretrained, str):
            print("this func is not realized")
            # logger = get_root_logger()
            # load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            generation_init_weights(
                self, init_type=self.init_type, init_gain=self.init_gain)
        else:
            raise TypeError("'pretrained' must be a str or None. "
                            f'But received {type(pretrained)}.')


@Discriminator.register_module()
class SimpleDiscriminator(BaseModule):

    def __init__(self, lamda_grl_layer=1.0):

        super(SimpleDiscriminator, self).__init__()

        self.domain_classifier = nn.Sequential()
        self.domain_classifier.add_module('fc1', nn.Linear(7*7, 1024))
        self.domain_classifier.add_module('relu1', nn.ReLU(True))
        self.domain_classifier.add_module('dpt1', nn.Dropout())
        self.domain_classifier.add_module('fc2', nn.Linear(1024, 1024))
        self.domain_classifier.add_module('relu2', nn.ReLU(True))
        self.domain_classifier.add_module('dpt2', nn.Dropout())
        self.domain_classifier.add_module('fc3', nn.Linear(1024, 2))
        self.init_type = 'normal'
        self.init_gain=0.02

        #self.grl_layer = GrlLayer(lamda_grl_layer)

    def forward(self, x):
        x = self.domain_classifier(x)

        return x

    def init_weights(self, pretrained=None):
        """Initialize weights for the model.

        Args:
            pretrained (str, optional): Path for pretrained weights. If given
                None, pretrained weights will not be loaded. Default: None.
        """
        if isinstance(pretrained, str):
            print("this func is not realized")
            # logger = get_root_logger()
            # load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            generation_init_weights(
                self, init_type=self.init_type, init_gain=self.init_gain)
        else:
            raise TypeError("'pretrained' must be a str or None. "
                            f'But received {type(pretrained)}.')

@Discriminator.register_module()
class InstanceDiscriminator(BaseModule):

    def __init__(self, num_stages, discriminator, lamda_grl_layer=1.0):

       super(InstanceDiscriminator, self).__init__()
       # self.s_max_pool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
       # self.t_max_pool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
       self.conv = nn.Conv2d(256,1,kernel_size=1,stride=1,padding=0,bias=False)
       self.dis = nn.ModuleList()
       for i in range(num_stages):
            self.dis.append(build_dicriminator(discriminator))

       self.grl_layer = GrlLayer(lamda_grl_layer)
       self.init_weights()

    def init_weights(self):
        for d in self.dis:
            d.init_weights()
        nn.init.kaiming_normal(self.conv.weight)

    def forward(self,source_f, target_f):

        #--------------------------#
        # source_f(list(3)->dict(2)): 3 means have 3 cascade rcnn stages
        #   source_f[0]: dict(2)->{"ROI_features","rois"}
        #    "roi_feats": [1024,256,7,7]
        # target_f(list(3)->dict(2)): 3 means have 3 cascade rcnn stages
        #   target_f[0]: dict(2)->{"ROI_features","rois"}
        #    "roi_feats": [1024,256,7,7]
        #--------------------------#
        loss_dict=dict()
        for i, features in enumerate(zip(source_f,target_f, self.dis)):
            source,target,d = features
            source_features = source["roi_feats"]
            target_features = target["roi_feats"]

            rsf = self.grl_layer(source_features)
            rtf = self.grl_layer(target_features)

            # [b,1,7,7],b=2048
            rsf = self.conv(rsf)
            rtf = self.conv(rtf)

            # [b,49],b=2048
            rsf = rsf.flatten(1)
            rtf = rtf.flatten(1)

            # source_d:[1024,49,2]
            soure_d = d(rsf)
            target_d = d(rtf)

            sdomain_label = torch.zeros(soure_d.size(0)).long().to(soure_d.device)
            err_s_domain = F.nll_loss(F.log_softmax(soure_d, dim=1), sdomain_label)
            tdomain_label = torch.ones(target_d.size(0)).long().to(target_d.device)
            err_t_domain = F.nll_loss(F.log_softmax(target_d, dim=1), tdomain_label)

            global_loss = pow(0.05,3-i) * (err_s_domain + err_t_domain)
            loss_dict[f"stage_loss_{i}"] = global_loss

        return loss_dict


    def forward_test1(self,source_f, target_f):
        from ..utils import proxy_a_distance
        #source_f, target_f

        for i, features in enumerate(zip(source_f, target_f, self.dis)):
            source, target, d = features
            source_features = source["roi_feats"]
            target_features = target["roi_feats"]

            # rsf = self.grl_layer(source_features)
            # rtf = self.grl_layer(target_features)
            #
            # # [b,1,7,7],b=2048
            # rsf = self.conv(rsf).reshape(-1,49)
            # rtf = self.conv(rtf).reshape(-1,49)
            #
            # rsf = rsf.detach().cpu().numpy()[:40]
            # rtf = rtf.detach().cpu().numpy()[:40]
            # # print("get into proxy_a_distance")
            # dis = proxy_a_distance(rsf, rtf)
            # print(dis)

            rsf = source_features.reshape(-1, 256 * 7 * 7).detach().cpu().numpy()[:200]
            rtf = target_features.reshape(-1, 256 * 7 * 7).detach().cpu().numpy()[:200]

            dis = proxy_a_distance(rsf, rtf)
            print(dis)
            # if dis<1.2:
            #     print(dis)


    def forward_test(self,source_f, target_f):
        from ..utils import proxy_a_distance
        #source_f, target_f

        for i, features in enumerate(zip(source_f, target_f, self.dis)):
            source, target, d = features
            source_features = source["roi_feats"]
            target_features = target["roi_feats"]

            # rsf = self.grl_layer(source_features)
            # rtf = self.grl_layer(target_features)
            #
            # # [b,1,7,7],b=2048
            # rsf = self.conv(rsf).reshape(-1,49)
            # rtf = self.conv(rtf).reshape(-1,49)
            #
            # rsf = rsf.detach().cpu().numpy()[:40]
            # rtf = rtf.detach().cpu().numpy()[:40]
            # # print("get into proxy_a_distance")
            # dis = proxy_a_distance(rsf, rtf)
            # print(dis)

            # rsf = source_features.reshape(-1, 256 * 7 * 7).detach().cpu().numpy()[:200]
            # rtf = target_features.reshape(-1, 256 * 7 * 7).detach().cpu().numpy()[:200]
            #
            # dis = proxy_a_distance(rsf, rtf)
            # print(dis)
            # if dis<1.2:
            #     print(dis)






@Discriminator.register_module()
class MultiLevelDiscriminator(BaseModule):

    def __init__(self,num_features, d, gan_loss,lamda_grl_layer=1.0):
        super(MultiLevelDiscriminator, self).__init__()
        # self.d1 = build_dicriminator(d1)
        # self.d2 = build_dicriminator(d2)
        # self.d3 = build_dicriminator(d3)
        self.num_features = num_features
        self.dis=nn.ModuleList()
        for i in range(num_features):
            self.dis.append(build_dicriminator(d))

        self.gan_loss = build_loss(gan_loss)
        self.grl_layer = GrlLayer(lamda_grl_layer)
        self.init_type='normal'
        self.init_gain = 0.02
        self.init_weights()

    def forward(self,source_feature, target_feature):

        assert isinstance(source_feature, (list,tuple)), "source_feature is not a list"
        assert isinstance(target_feature, (list,tuple)), "target_feature is not a list"
        assert len(source_feature) == self.num_features or len(target_feature) == self.num_features,"num features is not matched with source " \
                                                                                                    "features or target features"

        assert len(source_feature)==len(target_feature),"len(source feature) is not equal to len(target feature)"
        self.grl_layer.set_lambda(1.0,device=source_feature[0].device)
        # loss_d1 = dict()
        # loss_d2 = dict()
        # loss_d3 = dict()
        # # loss_d =dict()
        # source_grl
        source_grl = []
        for f,dis in zip(source_feature,self.dis):
            source_grl.append(dis(self.grl_layer(f)))
        # source_1 = self.d1(self.grl_layer(source_feature[0]))
        # source_2 = self.d2(self.grl_layer(source_feature[1]))
        # source_3 = self.d3(self.grl_layer(source_feature[2]))
        #
        target_grl=[]
        for f,dis in zip(target_feature,self.dis):
            target_grl.append(dis(self.grl_layer(f)))
        #target_grl = [self.dis(self.grl_layer(f)) for f in target_feature]
        # target_1 = self.d1(self.grl_layer(target_feature[0]))
        # target_2 = self.d2(self.grl_layer(target_feature[1]))
        # target_3 = self.d3(self.grl_layer(target_feature[2]))


        # source_1 = self.gan_loss(self.d1(source),  True)

        # source_real =  self.gan_loss(source_1, True)
        # loss_1 = (source_real+source_loss1)*0.5
        losses = dict()
        for i, ss in enumerate(zip(source_grl, target_grl)):
            s=ss[0]
            t=ss[1]
            s1_loss = self.gan_loss(s, True)
            t1_loss = self.gan_loss(t, False)

            losses[f"imgda_d{i}_loss"] = s1_loss+t1_loss
            #losses[f"imgda_target{i}_loss"] = t1_loss


        return losses


    def init_weights(self,pretrained=None):
        """Initialize weights for the model.

        Args:
            pretrained (str, optional): Path for pretrained weights. If given
                None, pretrained weights will not be loaded. Default: None.
        """
        if isinstance(pretrained, str):
            print("this func is not realized")
            # logger = get_root_logger()
            # load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            generation_init_weights(
                self, init_type=self.init_type, init_gain=self.init_gain)
        else:
            raise TypeError("'pretrained' must be a str or None. "
                            f'But received {type(pretrained)}.')


    def _parse_losses(self, losses):
        """Parse the raw outputs (losses) of the network.

        Args:
            losses (dict): Raw output of the network, which usually contain
                losses and other necessary information.

        Returns:
            tuple[Tensor, dict]: (loss, log_vars), loss is the loss tensor \
                which may be a weighted sum of all losses, log_vars contains \
                all the variables to be sent to the logger.
        """
        log_vars = OrderedDict()
        for loss_name, loss_value in losses.items():
            if isinstance(loss_value, torch.Tensor):
                log_vars[loss_name] = loss_value.mean()
            elif isinstance(loss_value, list):
                log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
            # Allow setting None for some loss item.
            # This is to support dynamic loss module, where the loss is
            # calculated with a fixed frequency.
            elif loss_value is None:
                continue
            else:
                raise TypeError(
                    f'{loss_name} is not a tensor or list of tensors')

        # Note that you have to add 'loss' in name of the items that will be
        # included in back propagation.
        loss = sum(_value for _key, _value in log_vars.items()
                   if 'loss' in _key)

        log_vars['loss'] = loss
        for loss_name, loss_value in log_vars.items():
            # reduce loss when distributed training
            if dist.is_available() and dist.is_initialized():
                loss_value = loss_value.data.clone()
                dist.all_reduce(loss_value.div_(dist.get_world_size()))
            log_vars[loss_name] = loss_value.item()

        return loss, log_vars

