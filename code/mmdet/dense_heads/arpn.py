# Copyright (c) OpenMMLab. All rights reserved.
import copy
from typing import List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.ops import nms
from mmengine.structures import InstanceData
from torch import Tensor

from mmdet.registry import MODELS
from mmdet.utils import ConfigType, InstanceList, MultiConfig, OptInstanceList
from .guided_anchor_head import GuidedAnchorHead


@MODELS.register_module()
class APNHead(GuidedAnchorHead):
    def __init__(self,
                 in_channels: int,
                 num_classes: int = 1,
                 init_cfg: dict = dict(
                     type='Normal',
                     layer='Conv2d',
                     std=0.01,
                     override=dict(
                         type='Normal',
                         name='conv_loc',
                         std=0.01,
                         bias_prob=0.01)),
                 **kwargs) -> None:
        super().__init__(
            num_classes=num_classes,
            in_channels=in_channels,
            init_cfg=init_cfg,
            **kwargs)

    def _init_layers(self) -> None:
        """Initialize layers of the head."""
        self.rpn_conv = nn.Conv2d(
            self.in_channels, self.feat_channels, 3, padding=1)
        # 复杂度预测模块 -> 输出 anchor density map
        self.density_conv = nn.Sequential(
            nn.Conv2d(self.feat_channels, self.feat_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.feat_channels, 1, 1),
            nn.Sigmoid()
        )
        super(APNHead, self)._init_layers()

    def forward_single(self, x: torch.Tensor, return_density: bool = False):
        """Forward for a single scale level.
        Args:
            return_density (bool): 是否返回 density_map（预测阶段使用）
        """
        x = self.rpn_conv(x)
        x = F.relu(x, inplace=True)
        density_map = self.density_conv(x)  # (N,1,H,W)
        cls_score, bbox_pred, shape_pred, loc_pred = super().forward_single(x)

        if return_density:
            return cls_score, bbox_pred, shape_pred, loc_pred, density_map
        else:
            return cls_score, bbox_pred, shape_pred, loc_pred

    def loss_by_feat(
            self,
            cls_scores,
            bbox_preds,
            shape_preds,
            loc_preds,
            batch_gt_instances,
            batch_img_metas,
            batch_gt_instances_ignore=None,
            **kwargs):  # **kwargs 接收多余参数
        """计算训练损失"""
        losses = super().loss_by_feat(
            cls_scores,
            bbox_preds,
            shape_preds,
            loc_preds,
            batch_gt_instances,
            batch_img_metas,
            batch_gt_instances_ignore=batch_gt_instances_ignore
        )
        return dict(
            loss_rpn_cls=losses['loss_cls'],
            loss_rpn_bbox=losses['loss_bbox'],
            loss_anchor_shape=losses['loss_shape'],
            loss_anchor_loc=losses['loss_loc']
        )

    def _predict_by_feat_single(self,
                                cls_scores,
                                bbox_preds,
                                mlvl_anchors,
                                mlvl_masks,
                                img_meta,
                                cfg,
                                rescale=False,
                                density_maps=None):
        """预测阶段，结合 density_map 做动态锚点筛选"""
        cfg = self.test_cfg if cfg is None else cfg
        cfg = copy.deepcopy(cfg)

        mlvl_proposals = []
        for idx in range(len(cls_scores)):
            rpn_cls_score = cls_scores[idx]
            rpn_bbox_pred = bbox_preds[idx]
            anchors = mlvl_anchors[idx]
            mask = mlvl_masks[idx]
            density_map = density_maps[idx] if density_maps is not None else None

            if mask.sum() == 0:
                continue

            rpn_cls_score = rpn_cls_score.permute(1, 2, 0)
            if self.use_sigmoid_cls:
                rpn_cls_score = rpn_cls_score.reshape(-1)
                scores = rpn_cls_score.sigmoid()
            else:
                rpn_cls_score = rpn_cls_score.reshape(-1, 2)
                scores = rpn_cls_score.softmax(dim=1)[:, :-1]

            scores = scores[mask]
            rpn_bbox_pred = rpn_bbox_pred.permute(1, 2, 0).reshape(-1, 4)[mask, :]

            # 动态锚点控制
            if density_map is not None:
                density = density_map.permute(0, 2, 3, 1).reshape(-1)[mask]
                scores = scores * density  # 融合 density_map

            if cfg.nms_pre > 0 and scores.shape[0] > cfg.nms_pre:
                _, topk_inds = scores.topk(cfg.nms_pre)
                rpn_bbox_pred = rpn_bbox_pred[topk_inds, :]
                anchors = anchors[topk_inds, :]
                scores = scores[topk_inds]

            proposals = self.bbox_coder.decode(
                anchors, rpn_bbox_pred, max_shape=img_meta['img_shape'])

            if cfg.min_bbox_size >= 0:
                w = proposals[:, 2] - proposals[:, 0]
                h = proposals[:, 3] - proposals[:, 1]
                valid_mask = (w > cfg.min_bbox_size) & (h > cfg.min_bbox_size)
                proposals = proposals[valid_mask]
                scores = scores[valid_mask]

            proposals = torch.cat([proposals, scores[:, None]], dim=1)
            proposals, _ = nms(proposals[:, :4], proposals[:, -1],
                               cfg.nms.iou_threshold)
            proposals = proposals[:cfg.nms_post, :]
            mlvl_proposals.append(proposals)

        proposals = torch.cat(mlvl_proposals, 0)
        scores = proposals[:, 4]
        num = min(cfg.max_per_img, proposals.shape[0])
        _, topk_inds = scores.topk(num)
        proposals = proposals[topk_inds, :]

        bboxes = proposals[:, :-1]
        scores = proposals[:, -1]
        if rescale:
            bboxes /= bboxes.new_tensor(img_meta['scale_factor']).repeat((1, 2))

        results = InstanceData()
        results.bboxes = bboxes
        results.scores = scores
        results.labels = scores.new_zeros(scores.size(0), dtype=torch.long)
        return results