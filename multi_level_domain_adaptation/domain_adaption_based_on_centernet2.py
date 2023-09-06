import torch.nn as nn
from mmdet.models.detectors.base import BaseDetector
from ..builder import DETECTORS, build_backbone, build_head, build_neck,build_dicriminator
from mmcv.parallel import  MMDistributedDataParallel
from mmcv.runner import  auto_fp16
from torch.nn.utils import clip_grad
from torch.nn.parallel.distributed import _find_tensors

@DETECTORS.register_module()
class DaCenternet2(BaseDetector):


    def __init__(self,
                 backbone,
                 neck=None,
                 rpn_head=None,
                 roi_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None,
                 img_da=None,
                 instance_da=None,
                 ):
        super(DaCenternet2, self).__init__(init_cfg)
        backbone.pretrained = pretrained
        self.detector=nn.ModuleDict()

        if backbone is not  None:
            self.detector["backbone"]= build_backbone(backbone)#self.backbone = nn.Sequential(

        if neck is not None:
            self.detector["neck"] = build_neck(neck)

        if rpn_head is not None:
            rpn_train_cfg = train_cfg.rpn if train_cfg is not None else None
            rpn_head_ = rpn_head.copy()
            rpn_head_.update(train_cfg=rpn_train_cfg, test_cfg=test_cfg.rpn)
            self.detector["rpn_head"]=build_head(rpn_head_)
        if roi_head is not None:
            rcnn_train_cfg = train_cfg.rcnn if train_cfg is not None else None
            roi_head_ = roi_head.copy()
            roi_head_.update(train_cfg=rcnn_train_cfg)
            roi_head_.update(test_cfg=test_cfg.rcnn)
            roi_head_.pretrained = pretrained
            # self.detector.update(roi_head=build_head(roi_head_))
            self.detector["roi_head"] =build_head(roi_head_)
            #self.roi_head = build_head(roi_head_)
        self.img_da=None
        if img_da is not None:
            self.img_da = build_dicriminator(img_da)

        self.instance_da=None
        if instance_da is not  None:
            self.instance_da = build_dicriminator(instance_da)

        optimizer_config=dict(grad_clip=dict(max_norm=35, norm_type=2))
        self.optimize_tools = OptimizerHook(**optimizer_config)

        # i do not know how to use it next it
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self._init()


    def _init(self):
        # for m in self.backbone:
        #     if hasattr(m, 'init_weights'):
        #         m.init_weights()

        for key, m in self.detector.items():
            if hasattr(m, 'init_weights'):
                m.init_weights()

        if self.instance_da:
            self.instance_da.init_weights()

    @property
    def with_rpn(self):
        """bool: whether the detector has RPN"""
        return hasattr(self.get_module(self.detector), 'rpn_head') and self.get_module(self.detector)["rpn_head"] is not None

    @property
    def with_roi_head(self):
        """bool: whether the detector has a RoI head"""
        return hasattr(self.get_module(self.detector), 'roi_head') and self.get_module(self.detector)["roi_head"]is not None

    @property
    def with_bbox(self):
        """bool: whether the detector has a bbox head"""
        return ((hasattr(self.get_module(self.detector), 'roi_head') and self.get_module(self.detector)["roi_head"].with_bbox)
                or (hasattr(self, 'bbox_head') and self.bbox_head is not None))

    def extract_feat(self, img):
        """Directly extract features from the backbone+neck."""

        detector = self.get_module(self.detector)

        x = detector["backbone"](img)

        if detector["neck"] is not None:
            x = detector["neck"](x)

        return x

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None,
                      **kwargs):
        """
        Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.

            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.

            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.

            gt_labels (list[Tensor]): class indices corresponding to each box

            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.

            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.

            proposals : override rpn proposals with custom proposals. Use when
                `with_rpn` is False.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        x = self.extract_feat(img)

        losses = dict()

        # RPN forward and loss
        if self.with_rpn:
            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)

            # ------------------------------------------------------#
            # 得到RPN loss 和 proposal_list
            # ------------------------------------------------------#
            detector=self.get_module(self.detector)
            rpn_losses, proposal_list = detector["rpn_head"].forward_train(
                x,
                img_metas,
                gt_bboxes,
                gt_labels=None,
                gt_bboxes_ignore=gt_bboxes_ignore,
                proposal_cfg=proposal_cfg)
            losses.update(rpn_losses)
        else:
            proposal_list = proposals
        # 这里应该对proposal做一个聚类操作
        # -----------------------------------------------------------------#
        # 输入 x:特征图，image_metas: 图像信息， proposal_list: 建议框[n,4] or [n,5]
        #   gt_bboxes: 图像的真是bbox,gt_labels:图想的真是label，
        #   设置需要忽略的box，gt_bboxes_ignore, 输入的mask，gt_masks,
        # 输出：
        #    roi_loss
        # -----------------------------------------------------------------#
        roi_losses =detector["roi_head"].forward_train(x, img_metas, proposal_list,
                                                 gt_bboxes, gt_labels,
                                                 gt_bboxes_ignore, gt_masks,
                                                 **kwargs)
        losses.update(roi_losses)

        return losses

    def aug_test(self, imgs, img_metas, rescale=False):
        """Test with augmentations.

        If rescale is False, then returned bboxes and masks will fit the scale
        of imgs[0].
        """
        x = self.extract_feats(imgs)
        proposal_list = self.rpn_head.aug_test_rpn(x, img_metas)
        return self.roi_head.aug_test(
            x, proposal_list, img_metas, rescale=rescale)

    def simple_test(self, img, img_metas, proposals=None, rescale=False):
        """Test without augmentation."""
        detector = self.get_module(self.detector)
        assert self.with_bbox, 'Bbox head must be implemented.'
        x = self.extract_feat(img)
        if proposals is None:
            proposal_list = detector["rpn_head"].simple_test_rpn(x, img_metas,proposals)
        else:
            proposal_list = proposals
        #-------------------------------------------#
        # return proposal_list, 是为了检查proposal list
        #-------------------------------------------#
        return detector["roi_head"].simple_test(
            x, proposal_list, img_metas, rescale=rescale)


    def forward_test(self, imgs, img_metas, **kwargs):
        """
        Args:
            imgs (List[Tensor]): the outer list indicates test-time
                augmentations and inner Tensor should have a shape NxCxHxW,
                which contains all images in the batch.
            img_metas (List[List[dict]]): the outer list indicates test-time
                augs (multiscale, flip, etc.) and the inner list indicates
                images in a batch.
        """
        for var, name in [(imgs, 'imgs'), (img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError(f'{name} must be a list, but got {type(var)}')

        num_augs = len(imgs)
        if num_augs != len(img_metas):
            raise ValueError(f'num of augmentations ({len(imgs)}) '
                             f'!= num of image meta ({len(img_metas)})')

        # NOTE the batched image size information may be useful, e.g.
        # in DETR, this is needed for the construction of masks, which is
        # then used for the transformer_head.
        for img, img_meta in zip(imgs, img_metas):
            batch_size = len(img_meta)
            for img_id in range(batch_size):
                img_meta[img_id]['batch_input_shape'] = tuple(img.size()[-2:])

        if num_augs == 1:
            # proposals (List[List[Tensor]]): the outer list indicates
            # test-time augs (multiscale, flip, etc.) and the inner list
            # indicates images in a batch.
            # The Tensor should have a shape Px4, where P is the number of
            # proposals.
            if 'proposals' in kwargs:
                kwargs['proposals'] = kwargs['proposals'][0]
            return self.simple_test(imgs[0], img_metas[0], **kwargs)
        else:
            assert imgs[0].size(0) == 1, 'aug test does not support ' \
                                         'inference with batch size ' \
                                         f'{imgs[0].size(0)}'
            # TODO: support test augmentation for predefined proposals
            assert 'proposals' not in kwargs
            return self.aug_test(imgs, img_metas, **kwargs)

    def source_forward_train(self,img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None,
                      **kwargs):


        x = self.extract_feat(img)

        losses = dict()

        # RPN forward and loss
        if self.with_rpn:
            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)

            # ------------------------------------------------------#
            # 得到RPN loss 和 proposal_list
            # ------------------------------------------------------#
            detector = self.get_module(self.detector)
            rpn_losses, proposal_list = detector["rpn_head"].forward_train(
                x,
                img_metas,
                gt_bboxes,
                gt_labels=None,
                gt_bboxes_ignore=gt_bboxes_ignore,
                proposal_cfg=proposal_cfg)
            losses.update(rpn_losses)
        else:
            proposal_list = proposals

        roi_losses, rois = detector["roi_head"].forward_train(x, img_metas, proposal_list,
                                                        gt_bboxes, gt_labels,
                                                        gt_bboxes_ignore, gt_masks,
                                                        **kwargs)
        losses.update(roi_losses)

        return rois, x, losses

    def target_forward_train(self,
                       img,
                       img_metas,
                       gt_bboxes=None,
                       gt_labels=None,
                       gt_bboxes_ignore=None,
                       gt_masks=None,
                       proposals=None,
                       **kwargs):
        x = self.extract_feat(img)

        detector = self.get_module(self.detector)
        proposal_list = None
        # RPN forward and loss
        if self.with_rpn:
            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)

            # ------------------------------------------------------#
            # 得到RPN loss 和 proposal_list
            # ------------------------------------------------------#
            _, proposal_list = detector["rpn_head"].forward_train(
                x,
                img_metas,
                gt_bboxes,
                gt_labels,
                gt_bboxes_ignore,
                proposal_cfg)
            # losses.update(rpn_losses=rpn_losses)

            for p in proposal_list:
                if len(p) < 10:
                    continue

        roi_features = detector["roi_head"].forward_target(x, proposal_list, **kwargs)

        return  x, roi_features

    def forward_da_train(self,data,**kwargs):

        loss_dict=dict()

        source_data = data["source"]['img']
        source_img_metas = data["source"]['img_metas']
        source_gt_bboxes = data["source"]['gt_bboxes']
        source_gt_labels = data["source"]['gt_labels']

        target_data = data["target"]['img']

        # detector_loss(lits[3]->dict[2]: ROI_feature,RoIs)
        # source_rois(tuple(5):tensor) : multi-level features
        # detector_loss(dict(11)):

        source_rois, source_multi_level_feature,detector_loss = self.source_forward_train(source_data,
                                                                                        source_img_metas,
                                                                                        source_gt_bboxes,
                                                                                        gt_labels=source_gt_labels,
                                                                                        gt_bboxes_ignore=None)
        loss_dict["det_loss"]=detector_loss

        target_img_metas = data["target"]["img_metas"]
        tareget_multi_level_feature, target_rois = self.target_forward_train(target_data, img_metas=target_img_metas)

        #img_da = self.get_module(self.img_da)
        domain_adaptation_loss = self.img_da(source_multi_level_feature, tareget_multi_level_feature)
        loss_dict["imda_loss"] = domain_adaptation_loss
        # domain adaptation model

        #instance_da = self.get_module(self.instance_da)
        instance_loss = self.instance_da(source_rois, target_rois)
        loss_dict["instance_da_loss"] = instance_loss
        #loss_dict.update(domain_adaptation_loss)
        return loss_dict

    @auto_fp16(apply_to=('img',))
    def forward(self, data=None,img=None,img_metas=None, return_loss=True, **kwargs):
        """Calls either :func:`forward_train` or :func:`forward_test` depending
        on whether ``return_loss`` is ``True``.
        I hope the inputs is a  dict like{"soruce","tareget"}

        Note this setting will change the expected inputs. When
        ``return_loss=True``, img and img_meta are single-nested (i.e. Tensor
        and List[dict]), and when ``resturn_loss=False``, img and img_meta
        should be double nested (i.e.  List[Tensor], List[List[dict]]), with
        the outer list indicating test time augmentations.
        """

        # if torch.onnx.is_in_onnx_export():
        #     assert len(img_metas) == 1
        #     return self.onnx_export(img[0], img_metas[0])

        if return_loss:
            #return self.forward_train(**data, **kwargs)
            return self.forward_da_train(data, **kwargs)
        else:
            img = img
            img_metas=img_metas
            return self.forward_test(img, img_metas)


    def train_step(self, data, optimizer):
        """The iteration step during training.

        This method defines an iteration step during training, except for the
        back propagation and optimizer updating, which are done in an optimizer
        hook. Note that in some complicated cases or models, the whole process
        including back propagation and optimizer updating is also defined in
        this method, such as GAN.

        Args:
            data (dict): The output of dataloader.
            optimizer (:obj:`torch.optim.Optimizer` | dict): The optimizer of
                runner is passed to ``train_step()``. This argument is unused
                and reserved.

        Returns:
            dict: It should contain at least 3 keys: ``loss``, ``log_vars``, \
                ``num_samples``.

                - ``loss`` is a tensor for back propagation, which can be a \
                weighted sum of multiple losses.
                - ``log_vars`` contains all the variables to be sent to the
                logger.
                - ``num_samples`` indicates the batch size (when the model is \
                DDP, it means the batch size on each GPU), which is used for \
                averaging the logs.
        """
        losses = self(data)

        log_vars=dict()
        log_vars_1=dict()
        if self.img_da is not None:
            optimizer_imda = optimizer["img_da"]
            losses_imda, log_vars_1 = self._parse_losses(losses["imda_loss"])
            if isinstance(self.img_da, MMDistributedDataParallel):
                self.img_da.reducer.prepare_for_backward(_find_tensors(losses_imda))
            detector_grad_norm = self.optimize_tools.after_train_iter(self.img_da, optimizer_imda, losses_imda,retain_graph=True)

        log_vars_3 = dict()
        if self.instance_da is not None:
            optimizer_inda = optimizer["instance_da"]
            # losses_inda, log_vars = self._parse_losses(losses["inda"])
            losses_inda, log_vars_3 = self._parse_losses(losses["instance_da_loss"])
            if isinstance(self.instance_da, MMDistributedDataParallel):
                self.instance_da.reducer.prepare_for_backward(_find_tensors(losses_inda))
            detector_grad_norm = self.optimize_tools.after_train_iter(self.instance_da, optimizer_inda, losses_inda,retain_graph=True)

        optimizer_detector = optimizer["detector"]
        losses_det, log_vars_2 = self._parse_losses(losses["det_loss"])

        if isinstance(self.detector, MMDistributedDataParallel):
            self.detector.reducer.prepare_for_backward(_find_tensors(losses_det))
        detector_grad_norm = self.optimize_tools.after_train_iter(self.detector, optimizer_detector,losses_det)



        log_vars.update(**log_vars_1)
        log_vars.update(**log_vars_2)
        log_vars.update(**log_vars_3)

        outputs = dict(log_vars=log_vars, num_samples=len(data["source"]['img_metas']))

        return outputs
    def get_module(self, module):
        """Get `nn.ModuleDict` to fit the `MMDistributedDataParallel`
        interface.

        Args:
            module (MMDistributedDataParallel | nn.ModuleDict): The input
                module that needs processing.

        Returns:
            nn.ModuleDict: The ModuleDict of multiple networks.
        """
        if isinstance(module, MMDistributedDataParallel):
            return module.module
        return module

class OptimizerHook():

    def __init__(self, grad_clip=None):
        self.grad_clip = grad_clip

    def clip_grads(self, params):
        params = list(
            filter(lambda p: p.requires_grad and p.grad is not None, params))
        if len(params) > 0:
            return clip_grad.clip_grad_norm_(params, **self.grad_clip)

    def after_train_iter(self, model, optimizer,loss,retain_graph=False):

        optimizer.zero_grad()
        loss.backward(retain_graph=retain_graph)
        out=dict()
        if self.grad_clip is not None:
            grad_norm = self.clip_grads(model.parameters())
            # if grad_norm is not None:
            #     # Add grad norm to the logger
            #    out.update({'grad_norm': float(grad_norm)})
                                         #runner.outputs['num_samples'])
        optimizer.step()
        return float(grad_norm)

class OptimizerHook_2():

    def __init__(self, grad_clip=None):
        self.grad_clip = grad_clip

    def clip_grads(self, params):
        params = list(
            filter(lambda p: p.requires_grad and p.grad is not None, params))
        if len(params) > 0:
            return clip_grad.clip_grad_norm_(params, **self.grad_clip)

    def after_train_iter(self, runner):
        runner.optimizer.zero_grad()
        runner.outputs['loss'].backward()
        if self.grad_clip is not None:
            grad_norm = self.clip_grads(runner.model.parameters())
            if grad_norm is not None:
                # Add grad norm to the logger
                runner.log_buffer.update({'grad_norm': float(grad_norm)},
                                         runner.outputs['num_samples'])
        runner.optimizer.step()