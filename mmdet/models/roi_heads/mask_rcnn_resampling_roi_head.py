import torch

from mmdet.core import bbox2result, bbox2roi, build_assigner, build_sampler
from ..builder import HEADS, build_head, build_roi_extractor
from .base_roi_head import BaseRoIHead
from .test_mixins import BBoxTestMixin, MaskTestMixin


@HEADS.register_module()
class ResamplingRoIHead(BaseRoIHead, BBoxTestMixin, MaskTestMixin):
    """
    Mask rcnn head with consistent sampling for mask branch.
    """

    def init_assigner_sampler(self):
        """
        Initialize assigner and sampler
        """
        self.bbox_assigner = None
        self.bbox_sampler = None
        if self.train_cfg:
            self.bbox_assigner = build_assigner(self.train_cfg.assigner)
            self.bbox_sampler = build_sampler(
                self.train_cfg.sampler, context=self
            )
        
    def init_bbox_head(self, bbox_roi_extractor, bbox_head):
        """
        Initialize bbox_head
        """
        self.bbox_roi_extractor = build_roi_extractor(bbox_roi_extractor)
        self.bbox_head = build_head(bbox_head)

    def init_mask_head(self, mask_roi_extractor, mask_head):
        """
        Initialize mask_head
        """
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
        if self.with_bbox:
            self.bbox_roi_extractor.init_weights()
            self.bbox_head.init_weights()
        if self.mask_head:
            if not self.share_roi_extractor:
                self.mask_roi_extractor.init_weights()
            self.mask_head.init_weights()

    def _bbox_forward(self, x, rois):
        """
        Box head forward function used in both training and testing.
        """
        bbox_feats = self.bbox_roi_extractor(
            x[:self.bbox_roi_extractor.num_inputs], rois
        )
        if self.with_shared_head:
            bbox_feats = self.shared_head(bbox_feats)
        cls_score, bbox_pred = self.bbox_head(bbox_feats)

        bbox_results = dict(
            cls_score=cls_score, bbox_pred=bbox_pred, bbox_feats=bbox_feats
        )
        return bbox_results

    def _bbox_forward_train(self, x, sampling_results, gt_bboxes, gt_labels, img_metas):
        """
        Run forward function and calculate loss for box head in training
        """
        rois = bbox2roi([res.bboxes for res in sampling_results])
        bbox_results = self._bbox_forward(self, x, rois)

        bbox_targets = self.bbox_head.get_targets(
            sampling_results, gt_bboxes, gt_labels, self.train_cfg
        )
        loss_bbox = self.bbox_head.loss(
            bbox_results['cls_score'],
            bbox_results['bbox_pred'],
            rois,
            *bbox_targets
        )

        bbox_results.update(loss_bbox=loss_bbox, rois=rois, bbox_targets=bbox_targets)
        return bbox_results
    
    def _mask_forward(self, x, rois=None, pos_inds=None, bbox_feats=None):
        """
        Mask head forward function used in both training and testing
        """
        assert ((rois is not None) ^ (pos_inds is not None and bbox_feats is not None))

        if rois is not None:
            mask_feats = self.mask_roi_extractor(
                x[:self.mask_roi_extractor.num_inputs], rois
            )
            if self.with_shared_head:
                mask_feats = self.shared_head(mask_feats)
        else:
            assert bbox_feats is not None
            mask_feats = bbox_feats[pos_inds]

        mask_pred = self.mask_head(mask_feats)
        mask_results = dict(mask_pred=mask_pred, mask_feats=mask_feats)
        return mask_results

    def _mask_forward_train(self, x, sampling_results, bbox_feats, gt_masks,
                            img_metas):
        """
        Run forward function and calculate loss for mask head in training.
        """
        if not self.share_roi_extractor:
            pos_rois = bbox2roi([res.pos_bboxes for res in sampling_results])
            if pos_rois.shape[0] == 0:
                return dict(loss_mask=None)
            mask_results = self._mask_forward(x, pos_rois)
        else:
            pos_inds = []
            device = bbox_feats.device
            for res in sampling_results:
                pos_inds.append(
                    torch.ones(
                        res.pos_bboxes.shape[0],
                        device=device,
                        dtype=torch.uint8
                    )
                )
                pos_inds.append(
                    torch.zeros(
                        res.neg_bboxes.shape[0],
                        device=device,
                        dtype=torch.unit8
                    )
                )
                pos_inds = torch.cat(pos_inds)
                if pos_inds.shape[0] == 0:
                    return dict(loss_mask=None)
                mask_results = self._mask_forward(
                    x, pos_inds=pos_inds, bbox_feats=bbox_feats
                )
            
            mask_targets = self.mask_head.get_targets(sampling_results, gt_masks,
                                                    self.train_cfg)
            pos_labels = torch.cat([res.pos_gt_labels for res in sampling_results])
            loss_mask = self.mask_head.loss(mask_results['mask_pred'],
                                            mask_targets, pos_labels)
            mask_results.update(loss_mask=loss_mask, mask_targets=mask_targets)
            return mask_results
        
    def forward_train(self,
                    x,
                    img_metas,
                    proposal_list,
                    gt_bboxes,
                    gt_labels,
                    gt_bboxes_ignore=None,
                    gt_masks=None):
        """
        Args:
            x:
            img_metas:
            proposal:
            gt_bboxes:
            gt_labels:
            gt_bboxes_ignore:
            gt_masks:

        Returns:
            dict[str, Tensor]: a dictinary of loss components
        """
        # assign gts and sample proposals for bbox head
        if self.with_bbox or self.with_mask:
            num_imgs = len(img_metas)
            if gt_bboxes_ignore is None:
                gt_bboxes_ignore = [None for _ in range(num_imgs)]
            sampling_results = []
            proposal_pos_list = []
            for i in range(num_imgs):
                assign_result = self.bbox_assigner.assign(
                    proposal_list[i], gt_bboxes[i], gt_bboxes_ignore[i],
                    gt_labels[i]
                )
                sampling_result = self.bbox_sampler.sample(
                    assign_result, proposal_list[i],
                    gt_bboxes[i], gt_labels[i],
                    feats=[lvl_feat[i][None] for lvl_feat in x]
                )
                sampling_results.append(sampling_result)
                proposal_pos_list.append(sampling_result.pos_bboxes)
        
        losses = dict()
        # bbox head forward and loss
        if self.with_bbox:
            bbox_results = self._bbox_forward_train(x, sampling_results,
                                                gt_bboxes, gt_labels,
                                                img_metas)
            losses.update(bbox_results['loss_bbox'])
        
        # refine proposals
        #pos_is_gts = [res.pos_is_gt for res in sampling_result]
        labels = bbox_results['bbox_targets'][0]
        with torch.no_grad():
            roi_labels = torch.where(
                labels == self.bbox_head.num_classes,
                bbox_results['cls_score'][:, :-1].argmax(1),
                labels
            )
            proposal_refine_list = self.bbox_head.refine_pos_bboxes(
                bbox_results['rois'], roi_labels, labels,
                bbox_results['bbox_pred'], img_metas
            )
    
        # assign gts and sample proposals for mask head
        sampling_results = []
        for j in range(num_imgs):
            assign_result = self.bbox_assigner.assign(
                torch.cat((proposal_pos_list[j], proposal_refine_list[j]), dim=0),
                gt_bboxes[j], gt_bboxes_ignore[j],
                gt_labels[j]
            )
            sampling_result = self.bbox_sampler.sample(
                assign_result,
                torch.cat((proposal_pos_list[j], proposal_refine_list[j]), dim=0),
                gt_bboxes[j],
                gt_labels[j],
                feats=[lvl_feat[j][None] for lvl_feat in x]
            )
            sampling_results.append(sampling_result)
        # mask head forward and loss
        if self.with_mask:
            mask_results = self._mask_forward_train(x, sampling_results,
                                                    bbox_results['bbox_feats'],
                                                    gt_masks, img_metas)
            if mask_results['loss_mask'] is not None:
                losses.update(mask_results['loss_mask'])
            
        return losses

    def simple_test(self,
                    x,
                    proposal_list,
                    img_metas,
                    proposals=None,
                    rescale=False):
        """Test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'

        det_bboxes, det_labels = self.simple_test_bboxes(
            x, img_metas, proposal_list, self.test_cfg, rescale=rescale)
        bbox_results = [
            bbox2result(det_bboxes[i], det_labels[i],
                        self.bbox_head.num_classes)
            for i in range(len(det_bboxes))
        ]

        if not self.with_mask:
            return bbox_results
        else:
            segm_results = self.simple_test_mask(
                x, img_metas, det_bboxes, det_labels, rescale=rescale)
            return list(zip(bbox_results, segm_results))

    def aug_test(self, x, proposal_list, img_metas, rescale=False):
        """Test with augmentations.

        If rescale is False, then returned bboxes and masks will fit the scale
        of imgs[0].
        """
        # recompute feats to save memory
        det_bboxes, det_labels = self.aug_test_bboxes(x, img_metas,
                                                    proposal_list,
                                                    self.test_cfg)

        if rescale:
            _det_bboxes = det_bboxes
        else:
            _det_bboxes = det_bboxes.clone()
            _det_bboxes[:, :4] *= det_bboxes.new_tensor(
                img_metas[0][0]['scale_factor'])
        bbox_results = bbox2result(_det_bboxes, det_labels,
                                self.bbox_head.num_classes)

        # det_bboxes always keep the original scale
        if self.with_mask:
            segm_results = self.aug_test_mask(x, img_metas, det_bboxes,
                                            det_labels)
            return [(bbox_results, segm_results)]
        else:
            return [bbox_results]
    