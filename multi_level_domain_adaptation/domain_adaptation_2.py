
import torch
import torch.nn as nn
from mmdet.models.builder import build_dicriminator, build_detector, DETECTORS

from mmdet.models.multi_level_domain_adaptation.proposal_cluster import compute_cluster_targets
from mmdet.models.multi_level_domain_adaptation.MMD import MMD_loss
from mmcv.parallel import MMDistributedDataParallel

from torch.nn.parallel.distributed import _find_tensors
from mmdet.models.common.model_utils import set_requires_grad
import torch.distributed as dist
from collections import OrderedDict
from mmcv.runner import BaseModule
from torch.nn.utils import clip_grad


@DETECTORS.register_module()
class DomainAdaptation_2(BaseModule):
    '''
        change the discriminator method

    '''

    def __init__(self,
                 discriminators,
                 detector,
                 optimizer_config=None,
                 train_cfg=None,
                 test_cfg=None):
        super(DomainAdaptation_2, self).__init__()

        assert detector, "rpn is none"
        assert discriminators, "roi_head is none"
        # 每个里面都要有初始化函数
        self.detector = build_detector(detector)
        self.discriminators = build_dicriminator(discriminators)
        self.distance = MMD_loss()

        self.optimize_tools = OptimizerHook(**optimizer_config)

    def forward_train(self, data, optimizer):

        source_data = data["source"]['img']
        source_img_metas = data["source"]['img_metas']
        source_gt_bboxes = data["source"]['gt_bboxes']
        source_gt_labels = data["source"]['gt_labels']

        target_data      = data["target"]['img']
        loss_detector_var=dict()
        discriminators = self.get_module(self.discriminators)
        detector = self.get_module(self.detector)
        #optimizer_config = dict(grad_clip=dict(max_norm=5, norm_type=2))
        optimize_tools = self.optimize_tools
        all_loss = dict()
        ############################feature level adaptation##########################################

        ###################### source level detect loss
        # -----------------------------------------------------------#
        # 进入detector，求detect loss: dict(rpn_loss,roi loss)
        # [rois、roi_features]
        # source_rois->list[3],3 表示特征层的数量。 source_rois[0]: dict(“ROI_fetures”,“RoIs”),
        # source_rois[0]["ROI_fetures"]: 512,256,7,7; 其中每张图像上的bbox 数量是512，如果是两张就是512*2
        # RoIs:512,5
        # -----------------------------------------------------------#
        detector_loss, source_rois,source_multi_level_feature = detector.forward_train(source_data,
                                                    source_img_metas,
                                                    source_gt_bboxes,
                                                    gt_labels=source_gt_labels,
                                                    gt_bboxes_ignore=None)
        #losses.update(detector_loss)

        # -----------------------------------------------------------#
        # roi_feature 进行cluster，并提取出rois 所对应的特征
        # -----------------------------------------------------------#
        source_roi_feats = source_rois[0]["ROI_fetures"]
        source_roi_bbox = source_rois[0]["RoIs"]

        source_roi_feats = detector.cnonv(source_roi_feats)

        # try:
        #      source_roi_feats = source_roi_feats.view(len(source_img_metas),512,-1)
        # except:
        #     print(source_roi_feats.size())
        # source_roi_bbox = source_roi_bbox.view(len(source_img_metas),512,-1)
        source_rois_clusters=[]
        source_cluster_centers=[]
        for i in range(len(source_img_metas)):
            rois = source_roi_feats[source_roi_bbox[:, 0]==torch.tensor(i, dtype=source_roi_bbox.dtype)]
            roi_bbox = source_roi_bbox[source_roi_bbox[:, 0]==torch.tensor(i, dtype=source_roi_bbox.dtype)]
            if len(rois)<4:
                rois.repeat(4,1,1,1)
                roi_bbox.repeat(4,1)
                #print(len(rois))
            # elif len(rois) < 0:
            #     rois = torch.randn(4, 128, 7, 7)
            #     roi_bbox = torch.randn(4, 5)
            try:
                source_rois_cluster, source_cluster_center = compute_cluster_targets(proposals= roi_bbox ,
                                                                  features=rois)
            except:
                continue
            source_rois_clusters.append(source_rois_cluster)
            #source_cluster_centers.append(source_cluster_center)



        ###################### target level detect loss
        # -----------------------------------------------------------#
        # target 送入RPN head, 得到rpn_loss 和 rpn_proposal_list,
        # rpn_proposal_list 进行聚类
        # -----------------------------------------------------------#
        target_img_metas = data["target"]["img_metas"]
        tareget_multi_level_feature, target_rois = detector.forward_target(target_data, img_metas=target_img_metas)

        tt1=[]
        tt2=[]
        for i ,t in enumerate(target_rois):
            target_roi_feats = t["roi_feats"]
            target_roi_bbox = t["rois_bbox"]
            tt1.append(target_roi_feats)
            tt2.append(target_roi_bbox)
        target_roi_feats = torch.cat(tt1,axis=0)
        target_roi_bbox = torch.cat(tt2,axis=0)

        target_roi_feats = detector.cnonv(target_roi_feats)

        target_rois_clusters=[]
        target_cluster_centers=[]
        for i in range(len(target_img_metas)):
            rois = target_roi_feats[target_roi_bbox[:, 0] == torch.tensor(i, dtype=source_roi_bbox.dtype)]
            roi_bbox = target_roi_bbox[target_roi_bbox[:, 0] == torch.tensor(i, dtype=source_roi_bbox.dtype)]

            if len(rois)<4:
                rois.repeat(4,1,1,1)
                roi_bbox.repeat(4,1)
                #print(len( rois))
            elif len(rois)<1:
                continue
            try:
                rois, target_cluster_center = compute_cluster_targets(proposals=roi_bbox,
                                                                   features=rois)
            except:
                continue
            target_rois_clusters.append(rois)
            #target_cluster_centers.append(target_cluster_center)


        if (len(source_rois_clusters)>1) and (len(target_rois_clusters) >1):
            source_rois_clusters = torch.cat(source_rois_clusters, axis=0)
            target_rois_clusters = torch.cat(target_rois_clusters, axis=0)
            target_rois_clusters = detector.cnonv2(target_rois_clusters.squeeze().flatten(2).permute(0,2,1)).squeeze()
            source_rois_clusters = detector.cnonv2(source_rois_clusters.squeeze().flatten(2).permute(0,2,1)).squeeze()

        #
        # # -----------------------------------------------------------#
        # # source roi_feature 和 target roi feature 计算距离，让距离最小
        # # target_rois_cluster:[4,128,256,7,7]
        # # source_rois_cluster:[4,128,256,7,7]
        # # -----------------------------------------------------------#
            dis_loss = self.distance(source_rois_clusters, target_rois_clusters)
            detector_loss.update(dis_loss=dis_loss)
        #
        # ############################image level adaptation##################################################
        # # -----------------------------------------------------------#
        # # 进行域适应domain_adaptation_loss:dict{}
        # # 输入：
        # #         source_multi_level_feature: list[3], （batch,n,h,w）
        # #         tareget_multi_level_feature: list[3], （batch,n,h,w）
        # # 输出：
        # #         domain_adaptation_loss： {d1:loos_d1,d2:loss_d2...}
        # # -----------------------------------------------------------#
        set_requires_grad(self.discriminators, True)
        optimizer_dis = optimizer["discriminators"]
        #optimizer_dis.zero_grad()
        domain_adaptation_loss = discriminators(source_multi_level_feature, tareget_multi_level_feature)
        if isinstance(self.discriminators, MMDistributedDataParallel):
            self.discriminators.reducer.prepare_for_backward(_find_tensors(domain_adaptation_loss))
        #domain_adaptation_loss.backward(retain_graph=True)

        discri_grad_norm = optimize_tools.after_train_iter(self.discriminators,optimizer_dis,domain_adaptation_loss,retain_graph=True)
        #optimizer_dis.step()

        set_requires_grad(self.discriminators, False)
        optimizer_detector = optimizer["detector"]
        #optimizer_detector.zero_grad()
        loss_detector,loss_detector_var = self._parse_losses(detector_loss)
        if isinstance(self.detector, MMDistributedDataParallel):
            self.detector.reducer.prepare_for_backward(_find_tensors(loss_detector))

        #optimizer_detector.step()
        detector_grad_norm=optimize_tools.after_train_iter(self.detector,optimizer_detector,loss_detector)


        all_loss.update(loss_detector_var)
        all_loss.update(domain_adaptation_loss=domain_adaptation_loss.item())
        all_loss.update(dis_grad_norm = discri_grad_norm)
        all_loss.update(detector_grad_norm=detector_grad_norm)
        #all_loss.update(discri_loss=domain_adaptation_loss.item())


        # -----------------------------------------------------------#
        # 计算loss sef.parse_lose()
        # -----------------------------------------------------------#

        ###############################################################################################
        #loss_dict.update(dis_loss)

        return all_loss

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

        loss_var = self.forward_train(data,optimizer)

        outputs = dict(
            log_vars=loss_var, num_samples=len(data['source']['img_metas']))

        return outputs

    def forward(self,data,return_loss=True,**kwargs):

        #img = data["target"]
        if return_loss:
            pass
        else:
            detector = self.get_module(self.detector)
            return  detector.forward_test(data)



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

        loss = sum(_value for _key, _value in log_vars.items()
                   if 'loss' in _key)/len(log_vars)

        log_vars['loss'] = loss
        for loss_name, loss_value in log_vars.items():
            # reduce loss when distributed training
            if dist.is_available() and dist.is_initialized():
                loss_value = loss_value.data.clone()
                dist.all_reduce(loss_value.div_(dist.get_world_size()))
            log_vars[loss_name] = loss_value.item()

        return loss, log_vars


    def forward_test(self, data):
        detector = self.get_module(self.detector)

        detector.forward_test(**data)

        return

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