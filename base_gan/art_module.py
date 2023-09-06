# import torch
# from torch import nn
# from .grl_module import GRL
# from .discriminator import PatchDiscriminator
# from .gan_loss import GANLoss
#
#
#
#
# class AttentionMap():
#     pass
#
# class ARTModule():
#
#     def __init__(self):
#         super(ARTModule, self).__init__()
#         self.grl_module = GRL()
#         self.D1 = PatchDiscriminator()
#         self.adv_loss = GANLoss(type='GANLoss',
#                                 gan_type='lsgan',
#                                 real_label_val=1.0,
#                                 fake_label_val=0.0,
#                                 loss_weight=1.0)
#         self.attention_gen = AttentionMap()
#
#     def forward(self, target,source,attention_map):
#
#         A = target
#         B = source
#         ga = self.D1(self.grl(target))
#         gb = self.D1(self.grl(source))
#
#         #adv [b,1,h,w]
#         adv = self.adv_loss(ga, gb)
#
#         # attention_map->[b,1,h,w]
#         attention_map = self.attention_gen(attention_map)
#
#         loss = torch.mean(attention_map*adv)
#         return loss
#
#
