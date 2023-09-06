import torch
import torch.nn as nn
from mmdet.models.builder import build_dicriminator, build_detector, DETECTORS

from mmdet.models.multi_level_domain_adaptation.proposal_cluster import compute_cluster_targets
from mmdet.models.multi_level_domain_adaptation.MMD import MMD_loss
from mmcv.parallel import MMDistributedDataParallel


from mmdet.models.common.model_utils import set_requires_grad
import torch.distributed as dist
from collections import OrderedDict
from mmcv.runner import BaseModule
from torch.nn.utils import clip_grad
from mmdet.core.visualization import imshow_det_bboxes
import numpy as np
import mmcv
from torch.nn.parallel.distributed import _find_tensors


@DETECTORS.register_module()
class Only_detectorDomainAdaptation(BaseModule):

    def __init__(self,
                 discriminators,
                 detector,
                 optimizer_config=None,
                 train_cfg=None,
                 pretrained=None,
                 test_cfg=None):
        super(Only_detectorDomainAdaptation, self).__init__()

        assert detector, "rpn is none"
        assert discriminators, "roi_head is none"
        # 每个里面都要有初始化函数
        self.detector = build_detector(detector)
        #self.discriminators = build_dicriminator(discriminators)
        self.distance = MMD_loss()

        self.optimize_tools = OptimizerHook(**optimizer_config)

    def forward_train(self, data, optimizer):

        # source_data = data["source"]['img']
        # source_img_metas = data["source"]['img_metas']
        # source_gt_bboxes = data["source"]['gt_bboxes']
        # source_gt_labels = data["source"]['gt_labels']
        #
        # target_data      = data["target"]['img']

        source_data = data['img']
        source_img_metas = data['img_metas']
        source_gt_bboxes = data['gt_bboxes']
        source_gt_labels = data['gt_labels']

        #target_data      = data["target"]['img']

        loss_detector_var=dict()
        # discriminators = self.get_module(self.discriminators)
        #detector = self.get_module(self.detector)
        #optimizer_config = dict(grad_clip=dict(max_norm=5, norm_type=2))
        #optimize_tools = self.optimize_tools
        all_loss = dict()
        # detector_loss, source_rois,source_multi_level_feature = detector.forward_train(source_data,
        #                                             source_img_metas,
        #                                             source_gt_bboxes,
        #                                             gt_labels=source_gt_labels,
        #                                             gt_bboxes_ignore=None)
        # detector_loss= detector.forward_train(source_data,
        #                                             source_img_metas,
        #                                             source_gt_bboxes,
        #                                             gt_labels=source_gt_labels,
        #                                             gt_bboxes_ignore=None)
        outs = self.detector.train_step(data,None)


        # optimizer_detector = optimizer["detector"]
        #loss_detector,loss_detector_var = self._parse_losses(detector_loss)
        # if isinstance(self.detector, MMDistributedDataParallel):
        #     self.detector.reducer.prepare_for_backward(_find_tensors(loss_detector))
        # detector_grad_norm=optimize_tools.after_train_iter(self.detector,optimizer_detector,loss_detector)


        # all_loss.update(loss_detector_var)
        # all_loss.update(detector_grad_norm=detector_grad_norm)

        return outs

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


    def train_step(self,data, optimizer, running_status=None):

        #load data and then exp forward_tran
        outs = self.forward_train(data, optimizer)
        #loss_detector,loss_var = self.forward_train(data,optimizer)

        # outputs = dict(loss=loss_detector,
        #     log_vars=loss_var, num_samples=len(data['source']['img_metas']))

        return outs

    def forward(self,img,img_metas,return_loss=True,**kwargs):

        #img = data["target"]
        if return_loss:
            pass
        else:
            #detector = self.get_module(self.detector)
            return  self.detector.forward(img,img_metas,return_loss=False)



    def val_step(self,data, optimizer):

        detector = self.get_module(self.detector)
        #data = data["target"]
        outputs = detector.val_step(data)
        return outputs

    def _parse_losses(self, losses):
        """Parse the raw outputs (losses) of the network.

        Args:
            losses (dict): Raw output of the network, which usually contain
                losses and other necessary infomation.

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
            else:
                raise TypeError(
                    f'{loss_name} is not a tensor or list of tensors')
        #  calculatee loss in every gpu
        loss = sum(_value for _key, _value in log_vars.items()
                   if 'loss' in _key)

        log_vars['loss'] = loss

        # huizong evey type of loss in one gpu
        for loss_name, loss_value in log_vars.items():
            # reduce loss when distributed training
            if dist.is_available() and dist.is_initialized():
                loss_value = loss_value.data.clone()
                dist.all_reduce(loss_value.div_(dist.get_world_size()))
            log_vars[loss_name] = loss_value.item()

        return loss, log_vars


    # def forward_test(self, img, img_metas,**kwargs):
    #
    #     #detector = self.get_module(self.detector)
    #
    #     self.detector.forward_test([img], [img_metas])
    #
    #     return

    @property
    def with_mask(self):
        """bool: whether the detector has RPN"""
        return hasattr(self, 'mask') and self.rpn_head is not None

    def show_result(self,
                    img,
                    result,
                    score_thr=0.3,
                    bbox_color=(72, 101, 241),
                    text_color=(72, 101, 241),
                    mask_color=None,
                    thickness=2,
                    font_size=13,
                    win_name='',
                    show=False,
                    wait_time=0,
                    out_file=None):
        """Draw `result` over `img`.

        Args:
            img (str or Tensor): The image to be displayed.
            result (Tensor or tuple): The results to draw over `img`
                bbox_result or (bbox_result, segm_result).
            score_thr (float, optional): Minimum score of bboxes to be shown.
                Default: 0.3.
            bbox_color (str or tuple(int) or :obj:`Color`):Color of bbox lines.
               The tuple of color should be in BGR order. Default: 'green'
            text_color (str or tuple(int) or :obj:`Color`):Color of texts.
               The tuple of color should be in BGR order. Default: 'green'
            mask_color (None or str or tuple(int) or :obj:`Color`):
               Color of masks. The tuple of color should be in BGR order.
               Default: None
            thickness (int): Thickness of lines. Default: 2
            font_size (int): Font size of texts. Default: 13
            win_name (str): The window name. Default: ''
            wait_time (float): Value of waitKey param.
                Default: 0.
            show (bool): Whether to show the image.
                Default: False.
            out_file (str or None): The filename to write the image.
                Default: None.

        Returns:
            img (Tensor): Only if not `show` or `out_file`
        """
        img = mmcv.imread(img)
        img = img.copy()
        if isinstance(result, tuple):
            bbox_result, segm_result = result
            if isinstance(segm_result, tuple):
                segm_result = segm_result[0]  # ms rcnn
        else:
            bbox_result, segm_result = result, None
        bboxes = np.vstack(bbox_result)
        labels = [
            np.full(bbox.shape[0], i, dtype=np.int32)
            for i, bbox in enumerate(bbox_result)
        ]
        labels = np.concatenate(labels)
        # draw segmentation masks
        segms = None
        if segm_result is not None and len(labels) > 0:  # non empty
            segms = mmcv.concat_list(segm_result)
            if isinstance(segms[0], torch.Tensor):
                segms = torch.stack(segms, dim=0).detach().cpu().numpy()
            else:
                segms = np.stack(segms, axis=0)
        # if out_file specified, do not show image in window
        if out_file is not None:
            show = False
        # draw bounding boxes
        img = imshow_det_bboxes(
            img,
            bboxes,
            labels,
            segms,
            class_names=self.CLASSES,
            score_thr=score_thr,
            bbox_color=bbox_color,
            text_color=text_color,
            mask_color=mask_color,
            thickness=thickness,
            font_size=font_size,
            win_name=win_name,
            show=show,
            wait_time=wait_time,
            out_file=out_file)

        if not (show or out_file):
            return img

class OptimizerHook():

    def __init__(self, grad_clip=None):
        self.grad_clip = grad_clip

    def clip_grads(self, params):
        params = list(
            filter(lambda p: p.requires_grad and p.grad is not None, params))
        if len(params) > 0:
            return clip_grad.clip_grad_norm_(params, **self.grad_clip)

    def after_train_iter(self, model, optimizer,loss,retain_graph=False):

        # CALCULATE GRAD
        optimizer.zero_grad()
        loss.backward(retain_graph=retain_graph)
        out=dict()
        if self.grad_clip is not None:
            grad_norm = self.clip_grads(model.parameters())
            # if grad_norm is not None:
            #     # Add grad norm to the logger
            #    out.update({'grad_norm': float(grad_norm)})
                                     #runner.outputs['num_samples'])
        # UPDATE GRAD
        optimizer.step()
        return float(grad_norm)