import numpy as np
import torch
import torch.nn as nn
from mmcv.runner import BaseModule
from mmcv.cnn import build_conv_layer, build_norm_layer
from mmdet.models import BACKBONES

from ..utils.delivotr_encoder_layer import DeLiVoTrEncoderLayer


@BACKBONES.register_module()
class DeLiVoTrEncoder(BaseModule):
    """
    DeLiVoTr Encoder/ backbone
    Args:
        d_model (int): the feature dimension of the encoder layer
        enc_num_layers (int): Number of encoder layers in VoxFormer
        checkpoint_layers: layer IDs (0 to num_enc_layers - 1) to use checkpoint.
            Note: In PyTorch 1.8, checkpoint function seems not able to receive
            dict as parameters. Better to use PyTorch >= 1.9.
        region_shape (list[int]): the shape of region
        sparse_shape (list[int]): the shape of sparse voxels after voxelization
        deli_cfg (dict): the DeLiVoTr config
    """

    def __init__(
            self,
            d_model,
            enc_num_layers=6,
            checkpoint_layers=[],
            region_shape=(20, 20),
            sparse_shape=(512, 512, 1),
            deli_cfg=dict(
                enc_min_depth=4,
                enc_max_depth=8,
                enc_width_mult=2.0,
                dextra_dropout=0.1,
                dextra_proj=2,
                attn_dropout=0.1,
                dropout=0.1,
                act_dropout=0.0,
                ffn_dropout=0.1,
                enc_ffn_red=4,
                norm_type='ln',
                act_type='relu',
                normalize_before=True,
            ),
            init_cfg=None,
            num_attached_conv=4,
            conv_kwargs=[
                dict(kernel_size=3, dilation=1, padding=1, stride=1),
                dict(kernel_size=3, dilation=1, padding=1, stride=1),
                dict(kernel_size=3, dilation=1, padding=1, stride=1),
                dict(kernel_size=3, dilation=2, padding=2, stride=1),
            ],
            norm_cfg=dict(type='naiveSyncBN2d', eps=1e-3, momentum=0.01),
            conv_cfg=dict(type='Conv2d', bias=False),
            conv_in_channel=128,
            conv_out_channel=128,
            conv_shortcut=True,
    ):
        super(DeLiVoTrEncoder, self).__init__(init_cfg=init_cfg)

        self.d_model = d_model
        self.enc_num_layers = enc_num_layers
        self.checkpoint_layers = checkpoint_layers
        self.region_shape = region_shape
        self.sparse_shape = sparse_shape

        self.num_attached_conv = num_attached_conv
        self.conv_shortcut = conv_shortcut

        # parameters required for DeLiVoTr encoder layer from deli_cfg
        self.layers = nn.ModuleList([])
        assert deli_cfg['enc_min_depth'] < deli_cfg['enc_max_depth']

        dextra_depths = np.linspace(start=deli_cfg['enc_min_depth'],
                                    stop=deli_cfg['enc_max_depth'],
                                    num=enc_num_layers,
                                    dtype=np.int)

        depth_ratio = (deli_cfg['enc_max_depth'] * 1.0) / deli_cfg[
            'enc_min_depth']

        width_multipliers = np.linspace(start=deli_cfg['enc_width_mult'],
                                        stop=deli_cfg['enc_width_mult'] + (
                                                depth_ratio - 1.0),
                                        # subtraction by 1 for max==min case
                                        num=enc_num_layers,
                                        dtype=np.float
                                        )

        # DeLiVoTr Encoder Layers (Intra-Region and Inter-Region)
        self.layers.extend(
            [
                DeLiVoTrEncoderLayer(d_model,
                                     sparse_shape=sparse_shape,
                                     region_shape=region_shape,
                                     deli_lay_cfg=dict(
                                         width_multiplier=round(
                                             width_multipliers[idx], 3),
                                         dextra_depth=layer_i,
                                         dextra_dropout=deli_cfg[
                                             'dextra_dropout'],
                                         dextra_proj=deli_cfg['dextra_proj'],
                                         attn_dropout=deli_cfg['attn_dropout'],
                                         dropout=deli_cfg['dropout'],
                                         act_dropout=deli_cfg['act_dropout'],
                                         ffn_dropout=deli_cfg['ffn_dropout'],
                                         enc_ffn_red=deli_cfg['enc_ffn_red'],
                                         norm_type=deli_cfg['norm_type'],
                                         act_type=deli_cfg['act_type'],
                                         normalize_before=deli_cfg[
                                             'normalize_before'],
                                     ),
                                     )
                for idx, layer_i in enumerate(dextra_depths)
            ]
        )

        self._reset_parameters()

        # for BEV
        self.output_shape = list(sparse_shape[0:2])

        if num_attached_conv > 0:
            conv_list = []
            for i in range(num_attached_conv):

                if isinstance(conv_kwargs, dict):
                    conv_kwargs_i = conv_kwargs
                elif isinstance(conv_kwargs, list):
                    assert len(conv_kwargs) == num_attached_conv
                    conv_kwargs_i = conv_kwargs[i]

                if i > 0:
                    conv_in_channel = conv_out_channel
                conv = build_conv_layer(
                    conv_cfg,
                    in_channels=conv_in_channel,
                    out_channels=conv_out_channel,
                    **conv_kwargs_i,
                )

                if norm_cfg is None:
                    convnormrelu = nn.Sequential(
                        conv,
                        nn.ReLU(inplace=True)
                    )
                else:
                    convnormrelu = nn.Sequential(
                        conv,
                        build_norm_layer(norm_cfg, conv_out_channel)[1],
                        nn.ReLU(inplace=True)
                    )
                conv_list.append(convnormrelu)

            self.conv_layer = nn.ModuleList(conv_list)

    def _reset_parameters(self):
        for name, p in self.named_parameters():
            if p.dim() > 1 and 'scaler' not in name and 'tau' not in name:
                nn.init.xavier_uniform_(p)

    def forward(self, voxel_info):
        """ Forward of DeLiVoTr Encoder
        Args:
            voxel_info (dict): A dict containing information of voxels from
                VoxFormer Input Layer, as follows:
                'batch_win_inds_shift0': (M,),
                'coors_in_win_shift0': (M, 3),
                'voxel_feats': (M, d),
                'voxel_coors': (M, 1+3),
                'voxel_keep_inds': (M,),
                'voxel_drop_level_shift0': (M,),
                'flat2win_inds_shift0': {0:(,), 1:(,), 2:(,)},
                'pos_dict_shift0': {0:(n_win, m_tok, d), 1:(n_win, m_tok, d),
                                    2:(n_win, m_tok, d)},
                'key_mask_shift0': {0:(n_win, m_tok, d), 1:(n_win, m_tok, d),
                                    2:(n_win, m_tok, d)},
                'shuffle_inds': (N,),
                'win_inds_bzyx_interreg': (M, 1+3),
                'win_num_xy': (win_x, win_y)
        Returns:
            voxel_feat_win_batch, pos_emb_win_batch, key_padding_win_batch (
            Tensor) : voxel region feats of shape (B, num_win, d)
        """

        assert voxel_info['voxel_coors'].dtype == torch.int64, 'data type' \
                                                               ' of coors' \
                                                               ' should be' \
                                                               ' torch.int64!'

        # batch_size = voxel_info['voxel_coors'][:, 0].max().item() + 1
        voxel_feat = voxel_info['voxel_feats']
        ind_dict = voxel_info['flat2win_inds_shift0']
        key_mask_dict = voxel_info['key_mask_shift0']
        pos_dict = voxel_info['pos_dict_shift0']

        win_inds_bzyx_interreg = voxel_info['win_inds_bzyx_interreg']
        # (M, 1+3)
        win_num_xy = voxel_info['win_num_xy']  # [win_x, win_y]
        voxel_coors = voxel_info['voxel_coors']

        output = voxel_feat  # (M, d)

        for i, layer in enumerate(self.layers):
            output = layer(output, pos_dict, ind_dict,
                           key_mask_dict,
                           win_inds_bzyx_interreg=win_inds_bzyx_interreg,
                           win_num_xy=win_num_xy,
                           using_checkpoint=i in self.checkpoint_layers)
        # (M, d)

        # using BEV
        batch_size = voxel_info['voxel_coors'][:, 0].max().item() + 1
        output = self.recover_bev(output, voxel_coors, batch_size)

        if self.num_attached_conv > 0:
            for conv in self.conv_layer:
                temp = conv(output)
                if temp.shape == output.shape and self.conv_shortcut:
                    output = temp + output
                else:
                    output = temp

        output_list = []
        output_list.append(output)

        return output_list
        # list[(B, C, H, W)]

    def recover_bev(self, voxel_feat, coors, batch_size):
        '''
        Args:
            voxel_feat: shape=[N, C]
            coors: [N, 4]
            batch_size (int)
        Return:
            batch_canvas:, shape=[B, C, ny, nx]
        '''
        ny, nx = self.output_shape
        feat_dim = voxel_feat.shape[-1]

        batch_canvas = []
        for batch_itt in range(batch_size):
            # Create the canvas for this sample
            canvas = torch.zeros(
                feat_dim,
                nx * ny,
                dtype=voxel_feat.dtype,
                device=voxel_feat.device)

            # Only include non-empty pillars
            batch_mask = coors[:, 0] == batch_itt
            this_coors = coors[batch_mask, :]
            indices = this_coors[:, 2] * nx + this_coors[:, 3]
            indices = indices.type(torch.long)
            voxels = voxel_feat[batch_mask, :]  # [n, c]
            voxels = voxels.t()  # [c, n]

            canvas[:, indices] = voxels

            batch_canvas.append(canvas)

        batch_canvas = torch.stack(batch_canvas, 0)

        batch_canvas = batch_canvas.view(batch_size, feat_dim, ny, nx)

        return batch_canvas
