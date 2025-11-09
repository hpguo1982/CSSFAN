# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from mmcv.cnn import ConvModule
from mmcv.cnn.bricks.transformer import MultiheadAttention
from mmdet.registry import MODELS
from .fpn import FPN


class CSSAM(nn.Module):
    """CSSAM"""
    def __init__(self, in_channels, ksize=(3, 3), stride=2, padding=1):
        super().__init__()
        self.key_value_unfold = nn.Unfold(ksize, stride=stride, padding=padding)
        self.query_unfold = nn.Unfold((1, 1))
        self.query_fold = partial(F.fold, kernel_size=ksize, stride=stride, padding=padding)
        self.attn = nn.MultiheadAttention(embed_dim=in_channels, num_heads=8, batch_first=True)
        self.kernel_numel = ksize[0] * ksize[1]

    def forward(self, feat: torch.Tensor, src: torch.Tensor) -> torch.Tensor:
        """
        feat: 下采样后的低层特征 [B, C, H, W]
        src: 高层特征 [B, C, H', W']
        """
        b, c, h, w = src.shape

        # query: 高层特征
        query = self.query_unfold(src)  # [B, C*1*1, H'*W']
        query = query.permute(0, 2, 1)  # [B, H'*W', C]

        # key/value: 下采样低层特征
        kv = self.key_value_unfold(feat)  # [B, C*ksize*ksize, H'*W']
        kv = kv.permute(0, 2, 1)  # [B, H'*W', C*ksize*ksize]

        # 如果通道数不同，需要先线性映射，这里保持通道一致
        kv = kv[..., :c]  # 截取前C通道，保证和query一致

        # 注意力计算
        out, _ = self.attn(query=query, key=kv, value=kv)  # [B, H'*W', C]

        # fold回原尺寸
        out = out.permute(0, 2, 1)  # [B, C, H'*W']
        out = F.fold(out, output_size=(h, w), kernel_size=(1, 1))  # [B, C, H', W']

        # 与原高层特征相乘（保留原公式）
        return out * src



@MODELS.register_module()
class CSSFAN(FPN):
   
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 start_level=0,
                 end_level=-1,
                 add_extra_convs=False,
                 relu_before_extra_convs=False,
                 no_norm_on_lateral=False,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=None,
                 init_cfg=dict(type='Xavier', layer='Conv2d', distribution='uniform')):
        super(CSSFAN, self).__init__(
            in_channels,
            out_channels,
            num_outs,
            start_level,
            end_level,
            add_extra_convs,
            relu_before_extra_convs,
            no_norm_on_lateral,
            conv_cfg,
            norm_cfg,
            act_cfg,
            init_cfg=init_cfg)


        self.downsample_convs = nn.ModuleList()
        self.crap_list = nn.ModuleList()
        self.pafpn_convs = nn.ModuleList()
        for i in range(self.start_level + 1, self.backbone_end_level):
            self.downsample_convs.append(
                ConvModule(
                    out_channels,
                    out_channels,
                    3,
                    stride=2,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    inplace=False
                )
            )
            self.crap_list.append(CSSAM(out_channels))
            self.pafpn_convs.append(
                ConvModule(
                    out_channels,
                    out_channels,
                    3,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    inplace=False
                )
            )

    def forward(self, inputs):

        assert len(inputs) == len(self.in_channels)

        # 构建 lateral features
        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        # 自底向上融合
        used_backbone_levels = len(laterals)
        for i in range(0, used_backbone_levels - 1):
            down_feat = self.downsample_convs[i](laterals[i])
            laterals[i + 1] = self.crap_list[i](feat=down_feat, src=laterals[i + 1])

        # 输出
        outs = []
        outs.append(laterals[0])
        for i in range(1, used_backbone_levels):
            outs.append(self.pafpn_convs[i - 1](laterals[i]))

        # 添加 extra levels
        if self.num_outs > len(outs):
            for i in range(self.num_outs - used_backbone_levels):
                outs.append(F.max_pool2d(outs[-1], 1, stride=2))

        return tuple(outs)
