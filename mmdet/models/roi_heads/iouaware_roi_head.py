import torch
import torch.nn as nn

from mmdet.core import (bbox2result, bbox2roi, build_assigner, build_sampler,
                        bbox_mapping, merge_aug_bboxes, merge_aug_masks,
                        multiclass_nms)
from ..builder import HEADS, build_head, build_roi_extractor
from .base_roi_head import BaseRoIHead
from .test_mixins import BBoxTestMixin, MaskTestMixin


@HEADS.register_module()
class IoUAwareRoIHead(BaseRoIHead, BBoxTestMixin, MaskTestMixin):
    """
    balanced iou-aware roi head
    """
    def __init__(self,
                num_stages=2,
                bbox_roi_extractor=None,
                bbox_head=None,
                mask_roi_extractor=None,
                mask_head=None,
                shared_head=None,
                train_cfg=None,
                test_cfg=None):
        self.num_stages = num_stages
        super(IoUAwareRoIHead, self).__init__(
            bbox_roi_extractor=bbox_roi_extractor,
            bbox_head=bbox_head,
            mask_roi_extractor=mask_roi_extractor,
            mask_head=mask_head,
            shared_head=shared_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg
        )

    def init_assigner_sampler(self):
        """
        Initialize assigner and sampler.
        """
        self.bbox_assigner = []
        self.bbox_sampler = []
        if self.train_cfg is not None:
            for idx, rcnn_train_cfg in enumerate(self.train_cfg):
                self.bbox_assigner.append(
                    build_assigner(rcnn_train_cfg.assigner)
                )
                if idx > 0:
                    self.current_stage = 1
                self.bbox_sampler.append(
                    build_sampler(rcnn_train_cfg.sampler, context=self)
                )
    
    def init_bbox_head(self, bbox_roi_extractor, bbox_head):
        """
        Initialize bbox_head
        """
        self.bbox_roi_extractor = nn.ModuleList()
        self.bbox_head = nn.ModuleList()
        if not isinstance(bbox_roi_extractor, list):
            bbox_roi_extractor = [bbox_roi_extractor for _ in range(2)]
        if not isinstance(bbox_head, list):
            bbox_head = [bbox_head for _ in range(2)]
        assert len(bbox_roi_extractor) == len(bbox_head) == 2

        for roi_extractor, head in zip(bbox_roi_extractor, bbox_head):
            self.bbox_roi_extractor.append(build_roi_extractor(roi_extractor))
            self.bbox_head.append(build_head(head))

    def init_mask_head(self, mask_roi_extractor, mask_head):
        """Initialize ``mask_head``"""
        if mask_roi_extractor is not None:
            self.mask_roi_extractor = build_roi_extractor(mask_roi_extractor)
            self.share_roi_extractor = False
        else:
            self.share_roi_extractor = True
            self.mask_roi_extractor = self.bbox_roi_extractor
        self.mask_head = build_head(mask_head)

    def init_weights(self, pretrained):
        """
        Initialize the weights in head.
        """
        if self.with_shared_head:
            self.shared_head.init_weights(pretrained=pretrained)
        for i in range(2):
            if self.with_bbox:
                self.bbox_roi_extractor[i].init_weights()
                self.bbox_head[i].init_weights()

    def _bbox_forward(self, stage, x, rois):
        """
        Box head forward funciton used in both training and testing.
        """
        bbox_roi_extractor = self.bbox_roi_extractor[stage]
        bbox_head = self.bbox_head[stage]
        bbox_feats = bbox_roi_extractor(x[:bbox_roi_extractor.num_inputs], rois)
        if stage == 0:
            cls_score, bbox_pred = bbox_head(bbox_feats)
            bbox_results = dict(cls_score=cls_score, bbox_pred=bbox_pred, bbox_feats=bbox_feats)
        else:
            iou_pred = bbox_head(bbox_feats)
            bbox_results = dict(iou_pred=iou_pred)
        return bbox_results
        
    def _bbox_forward_train(self, stage, x, sampling_results, gt_bboxes, 
                            gt_labels, rcnn_train_cfg):
        """
        Run forward function and calculate loss for box head in training.
        """
        rois = bbox2roi([res.bboxes for res in sampling_results])
        bbox_results = self._bbox_forward(stage, x, rois)
        bbox_targets = self.bbox_head[stage].get_targets(
            sampling_results, gt_bboxes, gt_labels, rcnn_train_cfg)
        
        if stage == 0:
            loss_bbox = self.bbox_head[stage].loss(bbox_results['cls_score'],
                                                    bbox_results['bbox_pred'],
                                                    rois,
                                                    *bbox_targets)
        else:
            loss_bbox = self.bbox_head[stage].loss(bbox_results['iou_pred'], 
                                                    rois, 
                                                    *bbox_targets)

        bbox_results.update(loss_bbox=loss_bbox, rois=rois, bbox_targets=bbox_targets)
        return bbox_results

    def forward_train(self,
                        x,
                        img_metas,
                        proposal_list,
                        gt_bboxes,
                        gt_labels,
                        gt_bboxes_ignore=None,
                        gt_mask=None):
        losses = dict()
    # classification and bounding box regression
        self.current_stage = 0
        rcnn_train_cfg = self.train_cfg[0]
        #assign gts and sample proposals
        sampling_results = []
        sampling_results_iou = []
        bbox_assigner = self.bbox_assigner[0]
        bbox_sampler = self.bbox_sampler[0]
        bbox_assigner_iou = self.bbox_assigner[1]
        bbox_sampler_iou = self.bbox_sampler[1]
        num_imgs = len(img_metas)
        if gt_bboxes_ignore is None:
            gt_bboxes_ignore = [None for _ in range(num_imgs)]
        for j in range(num_imgs):
            assign_result = bbox_assigner.assign(
                proposal_list[j], gt_bboxes[j], gt_bboxes_ignore[j],
                gt_labels[j]
            )
            sampling_result = bbox_sampler.sample(
                assign_result,
                proposal_list[j],
                gt_bboxes[j],
                gt_labels[j],
                feats=[lvl_feat[j][None] for lvl_feat in x]
            )
            assign_result_iou = bbox_assigner_iou.assign(
                proposal_list[j], gt_bboxes[j], gt_bboxes_ignore[j],
                gt_labels[j]
            )
            sampling_result_iou = bbox_sampler_iou.sample(
                assign_result_iou,
                proposal_list[j],
                gt_bboxes[j],
                gt_labels[j],
                feats=[lvl_feat[j][None] for lvl_feat in x]
            )
            sampling_results.append(sampling_result)
            sampling_results_iou.append(sampling_result_iou)
        # bbox head forward and loss
        bbox_results = self._bbox_forward_train(0, x, sampling_results,
                                        gt_bboxes, gt_labels, rcnn_train_cfg)
        losses.update(bbox_results['loss_bbox'])

        #refine bboxes
        proposal_initial_list = [res.bboxes for res in sampling_results_iou]
        #"""
        rois = bbox2roi(proposal_initial_list)
        bbox_results = self._bbox_forward(0, x, rois)
        bbox_targets = self.bbox_head[1].get_targets(
            sampling_results_iou, gt_bboxes, gt_labels, rcnn_train_cfg
        )
        labels = bbox_targets[0]
        with torch.no_grad():
            roi_labels = torch.where(
                labels == self.bbox_head[0].num_classes,
                bbox_results['cls_score'][:, :-1].argmax(1),
                labels
            )
            proposal_regressed_list = self.bbox_head[0].refine_pos_bboxes(
                rois, roi_labels, labels,
                bbox_results['bbox_pred'], img_metas
            )

    # iou regresion
        #baseline1
        self.current_stage = 1
        rcnn_train_cfg = self.train_cfg[2]
        sampling_results_regressed = []
        bbox_assigner = self.bbox_assigner[2]
        bbox_sampler = self.bbox_sampler[2]
        for j in range(num_imgs):
            assign_result = bbox_assigner.assign(
                proposal_regressed_list[j], gt_bboxes[j], gt_bboxes_ignore[j],
                gt_labels[j]
            )
            sampling_result = bbox_sampler.sample(
                assign_result,
                proposal_regressed_list[j],
                gt_bboxes[j],
                gt_labels[j],
                feats=[lvl_feat[j][None] for lvl_feat in x]
            )
            sampling_results_regressed.append(sampling_result)
        bbox_results = self._bbox_forward_train(1, x, sampling_results_regressed,
                                        gt_bboxes, gt_labels, rcnn_train_cfg)
        for name, value in bbox_results['loss_bbox'].items():
            losses[f'regressed.{name}'] = value * 2.0
        
        #"""
        #baseline0
        rcnn_train_cfg = self.train_cfg[1]
        sampling_results_initial = []
        ## constraint pos_iou_thr
        #with torch.no_grad():
        #    pos_iou = torch.cat([res.pos_ious for res in sampling_results_regressed], 0)
        #    pos_iou_mean = pos_iou.mean()
        #    pos_iou_std = pos_iou.std()
        #    pos_iou_thr = pos_iou_mean - pos_iou_std
        #    with open('./work_dirs/iou-aware_rcnn_balance/iou-aware-balance1_baseline2/min_pos_iou.txt', 'a') as f:
        #        f.write(f'pos_iou_mean={pos_iou_mean:.3f}, pos_iou_std={pos_iou_std:.3f}, pos_iou_thr={pos_iou_thr:.3f}\n')
        #if pos_iou_thr > 0.5:
        #    rcnn_train_cfg.assigner.pos_iou_thr = pos_iou_thr
        #    bbox_assigner = build_assigner(rcnn_train_cfg.assigner)
        #else:
        #    bbox_assigner = self.bbox_assigner[1]
        ## constraint pos_iou_thr
        bbox_assigner = self.bbox_assigner[1]
        bbox_sampler = self.bbox_sampler[1]
        for j in range(num_imgs):
            assign_result = bbox_assigner.assign(
                proposal_initial_list[j], gt_bboxes[j], gt_bboxes_ignore[j],
                gt_labels[j]
            )
            sampling_result = bbox_sampler.sample(
                assign_result,
                proposal_initial_list[j],
                gt_bboxes[j],
                gt_labels[j],
                feats=[lvl_feat[j][None] for lvl_feat in x]
            )
            sampling_results_initial.append(sampling_result)
        bbox_results = self._bbox_forward_train(1, x, sampling_results_initial,
                                        gt_bboxes, gt_labels, rcnn_train_cfg)
        for name, value in bbox_results['loss_bbox'].items():
            losses[f'initial.{name}'] = value * 1.0
        #"""
        
        return losses

    def simple_test(self, x, proposal_list, img_metas, rescale=False):
        """
        Test without augmentation
        """
        num_imgs = len(proposal_list)
        img_shapes = tuple(meta['img_shape'] for meta in img_metas)
        ori_shapes = tuple(meta['ori_shape'] for meta in img_metas)
        scale_factors = tuple(meta['scale_factor'] for meta in img_metas)

        rcnn_test_cfg = self.test_cfg
        rois = bbox2roi(proposal_list)

        bbox_results = self._bbox_forward(0, x, rois)
        cls_score = bbox_results['cls_score']
        bbox_pred = bbox_results['bbox_pred']
        num_proposals_per_img = tuple(len(proposals) for proposals in proposal_list)
        rois = rois.split(num_proposals_per_img, 0)
        cls_score = cls_score.split(num_proposals_per_img, 0)
        bbox_pred = bbox_pred.split(num_proposals_per_img, 0)

        #1. IoU evaluation performed only on the N refined bounding boxes 
        # which is refined by the coordinate of max cls_score.
        #""" 
        bbox_label = [s[:, :-1].argmax(dim=1) for s in cls_score]
        rois_refine = torch.cat([
            self.bbox_head[0].regress_by_class(rois[j], bbox_label[j], 
                                            bbox_pred[j], img_metas[j])
            for j in range(num_imgs)
        ])

        bbox_results = self._bbox_forward(1, x, rois_refine)
        iou_pred = bbox_results['iou_pred']
        iou_pred = iou_pred.split(num_proposals_per_img, 0)
        #"""
        """
        # 2. IoU evaluation performed on the N*K detected bounding boxes.
        proposal_refine_list = []
        for j in range(num_imgs):
            bboxes = self.bbox_head[0].bbox_coder.decode(
                rois[j][:, 1:], bbox_pred[j], img_metas[j]['img_shape']
            ).view(-1, 4)
            proposal_refine_list.append(bboxes)
        rois_refine = bbox2roi(proposal_refine_list)

        bbox_results = self._bbox_forward(1, x, rois_refine)
        num_classes = self.bbox_head[1].num_classes
        if self.bbox_head[1].reg_class_agnostic:
            iou_pred = bbox_results['iou_pred'].view(-1, num_classes)
        #else:
        #    iou_pred = bbox_results['iou_pred'].view(-1, num_classes, num_classes)
        iou_pred = iou_pred.split(num_proposals_per_img, 0)
        """



        # apply bbox post_processing to each image individually
        det_bboxes = []
        det_labels = []
        for i in range(num_imgs):
            det_bbox, det_label = self.bbox_head[-1].get_bboxes(
                rois[i],
                cls_score[i],
                bbox_pred[i],
                iou_pred[i],
                img_shapes[i],
                scale_factors[i],
                rescale=rescale,
                cfg=rcnn_test_cfg
            )
            det_bboxes.append(det_bbox)
            det_labels.append(det_label)
        bbox_results = [bbox2result(det_bboxes[i], det_labels[i],
                        self.bbox_head[-1].num_classes)
                    for i in range(num_imgs)]
        return bbox_results

    def aug_tets(self, features, proposal_list, img_metas, rescale=False):
        """
        Test with augmentations.
        """
        rcnn_test_cfg = self.test_cfg
        aug_bboxes = []
        aug_scores = []
        for x, img_meta in zip(features, img_metas):
            # only one image in the batch
            img_shape = img_meta[0]['img_shape']
            scale_factor = img_meta[0]['scale_factor']
            flip = img_meta[0]['flip']
            flip_direction = img_meta[0]['flip_direction']

            proposals = bbox_mapping(proposal_list[0][:, :4], img_shape,
                                    scale_factor, flip, flip_direction)
            
            rois = bbox2roi([proposals])
            
            bbox_results = self._bbox_forward(0, x, rois)
            cls_score = bbox_results['cls_score']
            bbox_pred = bbox_results['bbox_pred']

            bbox_label = cls_score[:, :-1].argmax(dim=1)
            rois_refine = self.bbox_head[0].regress_by_class(
                rois, bbox_label, bbox_results['bbox_pred'],
                img_meta[0]
            )

            bbox_results = self._bbox_forward(1, x, rois_refine)
            iou_pred = bbox_results['iou_pred']

            bboxes, scores = self.bbox_head[-1].get_bboxes(
                rois, 
                cls_score,
                bbox_pred,
                iou_pred,
                img_shape,
                scale_factor,
                rescale=False,
                cfg=None
            )
            aug_bboxes.append(bboxes)
            aug_scores.append(scores)
        
        # after merging, bboxes will be rescaled to the original image size
        merged_bboxes, merged_scores = merge_aug_bboxes(
            aug_bboxes, aug_scores, img_metas, rcnn_test_cfg
        )
        det_bboxes, det_labels = multiclass_nms(merged_bboxes, merged_scores,
                                        rcnn_test_cfg.score_thr,
                                        rcnn_test_cfg.nms,
                                        rcnn_test_cfg.max_per_img)
        bbox_result = bbox2result(det_bboxes, det_labels,
                            self.bbox_head[-1].num_classes)
        
        return bbox_result
