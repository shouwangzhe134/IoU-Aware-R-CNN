import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair

from mmdet.core import (auto_fp16, build_bbox_coder, force_fp32, multi_apply,
                        multiclass_nms, multiclass_nms_iou)
from mmdet.models.builder import HEADS, build_loss
from mmdet.models.losses import accuracy


@HEADS.register_module()
class IoUAwareBBoxHead(nn.Module):
    """
    iou-aware roi head, with three fc layers for classification, 
    bounding box regression and iou regression respectively.
    """

    def __init__(self,
                with_avg_pool=False,
                roi_feat_size=7,
                in_channels=256,
                fc_out_channels=1024,
                num_iou_fcs=2,
                num_classes=80,
                bbox_coder=dict(
                     type='DeltaXYWHBBoxCoder',
                     target_means=[0., 0., 0., 0.],
                     target_stds=[0.1, 0.1, 0.2, 0.2]),
                reg_class_agnostic=False,
                reg_decoded_bbox=False,
                loss_iou=dict(
                    type='SmoothL1Loss', beta=1.0, loss_weight=1.0
                )):
        super(IoUAwareBBoxHead, self).__init__()
        self.with_avg_pool = with_avg_pool
        self.roi_feat_size = _pair(roi_feat_size)
        self.roi_feat_area = self.roi_feat_size[0] * self.roi_feat_size[1]
        self.in_channels = in_channels
        self.fc_out_channels = fc_out_channels
        self.num_iou_fcs = num_iou_fcs
        self.num_classes = num_classes
        self.reg_class_agnostic = reg_class_agnostic
        self.reg_decoded_bbox = reg_decoded_bbox
        self.fpn16_enabled = False

        self.bbox_coder = build_bbox_coder(bbox_coder)
        self.loss_iou = build_loss(loss_iou)

        if self.with_avg_pool:
            self.avg_pool = nn.AvgPool2d(self.roi_feat_size)

        self.iou_fcs, self.iou_last_dim = \
            self._add_fc_branch(self.num_iou_fcs)
        
        out_dim_iou = 1 if self.reg_class_agnostic else self.num_classes
        self.fc_iou = nn.Linear(self.iou_last_dim, out_dim_iou)

        self.relu = nn.ReLU(inplace=True)
    
    def _add_fc_branch(self, num_fcs):
        """
        Add the fc branch which consists of a sequential of fc layers.
        """
        in_channels = self.in_channels
        if not self.with_avg_pool:
            in_channels *= self.roi_feat_area
        last_layer_dim = in_channels
        branch_fcs = nn.ModuleList()
        if num_fcs > 0:
            for i in range(num_fcs):
                fc_in_chnnels = (in_channels if i == 0 else self.fc_out_channels)
                branch_fcs.append(nn.Linear(fc_in_chnnels, self.fc_out_channels))
            last_layer_dim = self.fc_out_channels
        return branch_fcs, last_layer_dim

    def init_weights(self):
        nn.init.normal_(self.fc_iou.weight, 0, 0.001)
        nn.init.constant_(self.fc_iou.bias, 0)
        for m in self.iou_fcs.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        if self.with_avg_pool:
            x = self.avg_pool(x)
        x = x.view(x.size(0), -1)

        for iou_fc in self.iou_fcs:
            x = self.relu(iou_fc(x))
        iou_pred = self.fc_iou(x)
        
        return iou_pred
    
    def get_targets(self,
                    sampling_results,
                    gt_labels,
                    gt_bboxes,
                    rcnn_train_cfg,
                    concat=True):
        pos_bboxes_list = [res.pos_bboxes for res in sampling_results]
        neg_bboxes_list = [res.neg_bboxes for res in sampling_results]
        pos_gt_labels_list = [res.pos_gt_labels for res in sampling_results]
        pos_gt_ious_list = [res.pos_ious for res in sampling_results]
        labels, iou_targets, iou_weights = multi_apply(
            self._get_target_single,
            pos_bboxes_list,
            neg_bboxes_list,
            pos_gt_labels_list,
            pos_gt_ious_list,
            cfg=rcnn_train_cfg
        )

        if concat:
            labels = torch.cat(labels, 0)
            iou_targets = torch.cat(iou_targets, 0)
            iou_weights = torch.cat(iou_weights, 0)
        return labels, iou_targets, iou_weights

    def _get_target_single(self, pos_bboxes, neg_bboxes,pos_gt_labels,
                            pos_ious, cfg):
        num_pos = pos_bboxes.size(0)
        num_neg = neg_bboxes.size(0)
        num_samples = num_pos + num_neg

        #with open('./debug.txt', 'a') as f:
        #    f.write(str(pos_ious) + '\n')

        labels = pos_bboxes.new_full((num_samples, ), self.num_classes, 
                                    dtype=torch.long)
        iou_targets = pos_bboxes.new_zeros(num_samples, 1)
        iou_weights = pos_bboxes.new_zeros(num_samples, 1)

        if num_pos > 0:
            labels[:num_pos] = pos_gt_labels
            iou_targets[:num_pos] = pos_ious
            iou_weights[:num_pos] = 1.0
        
        return labels, iou_targets, iou_weights

    @force_fp32(apply_to=('iou_pred', ))
    def loss(self,
        iou_pred,
        rois, 
        labels,
        iou_targets,
        iou_weights,
        reduction_override=None):
        losses = dict()
        if iou_pred is not None:
            pos_inds = (iou_weights.view(-1) > 0)
            # do not perform bounding box regression for BG anymore.
            if pos_inds.any():
                if self.reg_class_agnostic:
                    pos_iou_pred = iou_pred[pos_inds.type(torch.bool)]
                else:
                    pos_iou_pred = iou_pred[pos_inds.type(torch.bool), labels[pos_inds.type(torch.bool)]][:, None]

                losses['loss_iou'] = self.loss_iou(
                    pos_iou_pred,
                    iou_targets[pos_inds.type(torch.bool)],
                    iou_weights[pos_inds.type(torch.bool)],
                    avg_factor=iou_targets.size(0),# num = 64
                    #avg_factor=0.5*iou_targets.size(0),# num = 128
                    reduction_override=reduction_override
                )
            else:
                losses['loss_iou'] = iou_pred.sum() * 0
        return losses
    
    @force_fp32(apply_to=('cls_score', 'bbox_pred', 'iou_pred'))
    def get_bboxes(self,
                    rois,
                    cls_score,
                    bbox_pred,
                    iou_pred,
                    img_shape,
                    scale_factor,
                    rescale=False,
                    cfg=None):
        if isinstance(cls_score, list):
            cls_score = sum(cls_score) / float(len(cls_score))
        scores = F.softmax(cls_score, dim=1) if cls_score is not None else None

        if bbox_pred is not None:
            bboxes = self.bbox_coder.decode(
                rois[:, 1:], bbox_pred, max_shape=img_shape)
        else:
            bboxes = rois[:, 1:].clone()
            if img_shape is not None:
                bboxes[:, [0, 2]].clamp_(min=0, max=img_shape[1])
                bboxes[:, [1, 3]].clamp_(min=0, max=img_shape[0])
        if rescale and bboxes.size(0) > 0:
            if isinstance(scale_factor, float):
                bboxes /= scale_factor
            else:
                scale_factor = bboxes.new_tensor(scale_factor)
                bboxes = (bboxes.view(bboxes.size(0), -1, 4) / scale_factor).view(bboxes.size()[0], -1)
        
        if cfg is None:
            return bboxes, scores, iou_pred
        else:
            det_bboxes, det_labels = multiclass_nms_iou(bboxes, scores, iou_pred,
                                                cfg.score_thr, cfg.nms,
                                                cfg.max_per_img)
            return det_bboxes, det_labels
    