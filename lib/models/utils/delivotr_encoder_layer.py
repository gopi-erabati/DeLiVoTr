# using flat2win_v2 without voxel_drop_level
import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from ...ops.dynamic_scatter import DynamicScatterCustom
from lib.ops import flat2window_v2, window2flat_v2
from .dextra_unit import DExTraUnit
from .single_head_attention import SingleHeadAttention
from .helpers import get_norm_layer, get_activation_layer
from .nn_functions import get_weight_layer


class IntraRegionTransformerEncoder(nn.Module):
    """ Intra Region Transformer Encoder with

    Args:
        d_model (int): Dimension of model
    """

    def __init__(self, d_model, deli_lay_cfg=dict()):

        super().__init__()
        self.embed_dim = d_model
        width_multiplier = deli_lay_cfg['width_multiplier']
        dextra_depth = deli_lay_cfg['dextra_depth']
        dextra_dropout = deli_lay_cfg['dextra_dropout']
        dextra_proj = deli_lay_cfg['dextra_proj']
        attn_dropout = deli_lay_cfg['attn_dropout']
        dropout = deli_lay_cfg['dropout']
        act_dropout = deli_lay_cfg['act_dropout']
        ffn_dropout = deli_lay_cfg['ffn_dropout']
        enc_ffn_red = deli_lay_cfg['enc_ffn_red']
        norm_type = deli_lay_cfg['norm_type']
        act_type = deli_lay_cfg['act_type']
        normalize_before = deli_lay_cfg['normalize_before']
        assert d_model % dextra_proj == 0

        self.proj_dim = d_model // dextra_proj
        max_groups = 2 ** math.ceil(math.log(d_model // 32, 2))
        self.dextra_layer = DExTraUnit(in_features=self.embed_dim,
                                       in_proj_features=self.proj_dim,
                                       out_features=self.proj_dim,
                                       width_multiplier=width_multiplier,
                                       dextra_depth=dextra_depth,
                                       dextra_dropout=dextra_dropout,
                                       max_glt_groups=max_groups,
                                       act_type='gelu',
                                       use_bias=True,
                                       norm_type='ln',
                                       glt_shuffle=True,
                                       )

        self.self_attn = SingleHeadAttention(q_in_dim=self.proj_dim,
                                             k_in_dim=self.proj_dim,
                                             v_in_dim=self.proj_dim,
                                             proj_dim=self.proj_dim,
                                             out_dim=self.embed_dim,
                                             dropout=attn_dropout,
                                             bias=True)

        self.self_attn_layer_norm = get_norm_layer(name=norm_type,
                                                   out_features=self.embed_dim)
        self.dropout = dropout
        self.norm_fn = norm_type
        self.act_type = act_type
        self.activation_fn = get_activation_layer(name=act_type)
        self.activation_dropout = act_dropout
        self.normalize_before = normalize_before

        # Light-weight FFN
        self.ffn_dropout = ffn_dropout
        ffn_red_factor = enc_ffn_red
        assert self.embed_dim % ffn_red_factor == 0, '{}/{} should be a perfect divisor'.format(
            self.embed_dim,
            ffn_red_factor)
        light_ffn_dim = self.embed_dim // ffn_red_factor
        self.fc1 = get_weight_layer(name='linear',
                                    in_features=self.embed_dim,
                                    out_features=light_ffn_dim,
                                    use_bias=True)
        self.fc2 = get_weight_layer(name='linear',
                                    in_features=light_ffn_dim,
                                    out_features=self.embed_dim,
                                    use_bias=True)

        self.final_layer_norm = get_norm_layer(name=norm_type,
                                               out_features=self.embed_dim)

    def forward(
            self,
            src,
            pos_dict,
            ind_dict,
            key_padding_mask_dict,
    ):
        """ Forward of Intra Region TF Encoder
        Args:
            src (Tensor): Voxel Features from VFE of shape (M, d)
            pos_dict (dict): pos embeddings intra-region of shape
                {0:(n_win, m_tok, C), 1:(n_win, m_tok, C),
                 2:(n_win, m_tok, C)}
            ind_dict (dict): flat2win indices of shape
                {0:(,), 1:(,), 2:(,)}
            key_padding_mask_dict (dict): key padding mask intra-region
                of shape
                {0:(n_win, m_tok, C), 1:(n_win, m_tok, C),
                 2:(n_win, m_tok, C)}
        Returns:
            src (Tensor): Voxel Features of shape (M, d)
        """
        # add position embedding to the src
        out_feat_dict = {}

        feat_3d_dict = flat2window_v2(src, ind_dict)
        # {0: (n_win, m_tok, d), 1: (n_win, m_tok, d), 2: (n_win, m_tok, d)}

        for name in feat_3d_dict:
            #  [n, num_token, embed_dim]
            pos = pos_dict[name]

            feat_3d = feat_3d_dict[name]  # (n_win, m_tok, d)
            feat_3d = feat_3d.permute(1, 0, 2)  # (m_tok, n_win, d)

            if pos is not None:
                pos = pos.permute(1, 0, 2)  # (m_tok, n_win, d)
                assert pos.shape == feat_3d.shape, f'pos_shape: {pos.shape},' \
                                                   f' feat_shape:{feat_3d.shape}'
                feat_3d = feat_3d + pos  # (m_tok, n_win, d)

            out_feat_dict[name] = feat_3d.permute(1, 0, 2)
            # (n_win, m_tok, d)

        # out_feat_dict is {0:(n_win, m_tok, d), 1:(n_win, m_tok, d),
        # 2:(n_win, m_tok, d)}
        results = window2flat_v2(out_feat_dict, ind_dict)  # (M, d)

        residual = results
        if self.normalize_before:
            src = self.self_attn_layer_norm(results)

        # dextra for dim mul and red
        results = self.dextra_layer(results)  # (M, d0)

        # attention
        out_feat_dict = {}

        feat_3d_dict = flat2window_v2(results, ind_dict)
        # {0: (n_win, m_tok, d), 1: (n_win, m_tok, d), 2: (n_win, m_tok, d)}

        for name in feat_3d_dict:
            feat_3d = feat_3d_dict[name]  # (n_win, m_tok, d)
            feat_3d = feat_3d.permute(1, 0, 2)  # (m_tok, n_win, d)

            key_padding_mask = key_padding_mask_dict[name]  # (n_win, m_tok)

            out_feat_3d, attn_map = self.self_attn(
                query=feat_3d, key=feat_3d, value=feat_3d,
                key_padding_mask=key_padding_mask)
            out_feat_dict[name] = out_feat_3d.permute(1, 0, 2)
            # (n_win, m_tok, d)

        # out_feat_dict is {0:(n_win, m_tok, d), 1:(n_win, m_tok, d),
        # 2:(n_win, m_tok, d)}
        results = window2flat_v2(out_feat_dict, ind_dict)  # (M, d)

        # dropout and skip connection
        results = F.dropout(results, p=self.dropout, training=self.training)
        results = residual + results  # (M, d)

        if not self.normalize_before:
            results = self.self_attn_layer_norm(results)

        # Light-weight FFN
        residual = results
        if self.normalize_before:
            results = self.final_layer_norm(results)  # (M, d)
        results = self.activation_fn(self.fc1(results))  # (N, B, d/4)
        results = F.dropout(results, p=float(self.activation_dropout),
                            training=self.training)
        results = self.fc2(results)  # (N, B, d)
        results = F.dropout(results, p=self.ffn_dropout, training=self.training)
        results = residual + results  # (N, B, d)
        if not self.normalize_before:
            results = self.final_layer_norm(results)
        return results  # (M, d)


class VoxelFeatureRegionAggregation(nn.Module):
    """ Class to aggregate features in a region to facilitate Inter-Region
    Attention
    """

    def __init__(self):

        super().__init__()
        # DynamicScatter to get the mean of all voxels in a region
        self.scatter = DynamicScatterCustom(None, None, True)

    def forward(self,
                voxel_feat,
                win_inds_bzyx_interreg,
                win_num_xy):
        """ Forward of Intra Region Encoder
        Args:
            voxel_feat (Tensor): Voxel feats of shape (M, d)
            win_inds_bzyx_interreg (Tensor): Window indices of each voxel of shape (M, 4) order: b, z, y, x
            win_num_xy (list): number of windows in x and y directions
        Returns:
            voxel_feat_win_batch (Tensor): Voxel feats of shape (B, num_win_per_batch, d)
            pos_emb_win_batch (Tensor): Pos emb of shape (B, num_win_per_batch, d)
            key_padding_win_batch (Tensor): Key mask of shape (B, num_win_per_batch)
            flat2batch_inds (Tensor): flat2batch indices of shape (M', )
            voxel_coord_win (tensor): voxel coordinates of aggregates feats
                in windows of shape (M', )
        """
        # get mean of all voxels in a region to do inter region attention
        voxel_feat_win, voxel_coord_win = self.scatter(voxel_feat,
                                                       win_inds_bzyx_interreg)
        # (M', d) (M', 4)

        # get the feats, pos_emb and key_padding mask for inter-region attn
        voxel_feat_win_batch, pos_emb_win_batch, key_padding_win_batch, flat2batch_inds = \
            self.get_feat_posemb_keymask_intereg(voxel_feat_win,
                                                 voxel_coord_win, win_num_xy)
        # (B, num_win_per_batch, d), (B, num_win_per_batch, d),
        # (B, num_win_per_batch), (M', )

        return (voxel_feat_win_batch, pos_emb_win_batch,
                key_padding_win_batch, flat2batch_inds, voxel_coord_win)

    def get_feat_posemb_keymask_intereg(self, voxel_feat_win,
                                        voxel_coord_win, win_num_xy):
        # get the indices to convert flat to batch
        flat2batch_inds = self.get_flat2batch_inds(voxel_coord_win, win_num_xy)
        # (M',)

        # get the voxel feats in 3d shape for attention
        voxel_feat_win_batch = self.flat2batch(voxel_feat_win,
                                               voxel_coord_win,
                                               flat2batch_inds, win_num_xy)
        # (B, num_win_per_batch, d)

        # get the pos embedding for the new voxel coordinates
        # get the position embedding of window indices for inter-region
        # attention
        pos_emb_wins_interreg = self.get_pos_emb_wins_interreg(
            voxel_coord_win[:, 1:],  # no batch
            win_num_xy,
            voxel_feat_win.size(
                1),
            voxel_feat_win.dtype)
        # (M', d)
        # get the pos embedding in 3d shape for attention
        pos_emb_win_batch = self.flat2batch(pos_emb_wins_interreg,
                                            voxel_coord_win,
                                            flat2batch_inds,
                                            win_num_xy)
        # (B, num_win_per_batch, d)

        # get the key padding mask in 3d shape for attention
        key_padding_win_batch = self.get_key_padding_mask_intereg(
            voxel_coord_win, flat2batch_inds, win_num_xy)
        # (B, num_win_per_batch)

        return voxel_feat_win_batch, pos_emb_win_batch, \
            key_padding_win_batch, flat2batch_inds
        # (B, num_win_per_batch, d), (B, num_win_per_batch, d),
        # (B, num_win_per_batch), (M',)

    @torch.no_grad()
    def get_pos_emb_wins_interreg(self, voxel_coord_win, win_num_xy,
                                  feat_dim, dtype, normalize_pos=False,
                                  pos_temperature=10000):
        '''
        Args:
            voxel_coord_win: shape=[M', 3], order: z, y, x
            win_num_xy: Number of windows in X and Y axis (num_win_x,
            num_win_y)
        '''

        window_shape = win_num_xy
        if len(window_shape) == 2:
            ndim = 2
            win_x, win_y = window_shape
            win_z = 0
        elif window_shape[-1] == 1:
            ndim = 2
            win_x, win_y = window_shape[:2]
            win_z = 0
        else:
            win_x, win_y, win_z = window_shape
            ndim = 3

        assert voxel_coord_win.size(1) == 3
        z, y, x = voxel_coord_win[:, 0] - win_z / 2, voxel_coord_win[:, 1] - \
                  win_y / 2, voxel_coord_win[:, 2] - win_x / 2

        assert (x >= -win_x / 2 - 1e-4).all()
        assert (x <= win_x / 2 - 1 + 1e-4).all()

        if normalize_pos:
            x = x / win_x * 2 * 3.1415  # [-pi, pi]
            y = y / win_y * 2 * 3.1415  # [-pi, pi]
            z = z / win_z * 2 * 3.1415  # [-pi, pi]

        pos_length = feat_dim // ndim
        # [pos_length]
        inv_freq = torch.arange(
            pos_length, dtype=torch.float32, device=voxel_coord_win.device)
        inv_freq = pos_temperature ** (2 * (inv_freq // 2) / pos_length)

        # [num_tokens, pos_length]
        embed_x = x[:, None] / inv_freq[None, :]
        embed_y = y[:, None] / inv_freq[None, :]
        if ndim == 3:
            embed_z = z[:, None] / inv_freq[None, :]

        # [num_tokens, pos_length]
        embed_x = torch.stack([embed_x[:, ::2].sin(), embed_x[:, 1::2].cos()],
                              dim=-1).flatten(1)
        embed_y = torch.stack([embed_y[:, ::2].sin(), embed_y[:, 1::2].cos()],
                              dim=-1).flatten(1)
        if ndim == 3:
            embed_z = torch.stack(
                [embed_z[:, ::2].sin(), embed_z[:, 1::2].cos()],
                dim=-1).flatten(1)

        # [num_tokens, c]
        if ndim == 3:
            pos_embed_2d = torch.cat([embed_x, embed_y, embed_z], dim=-1).to(
                dtype)
        else:
            pos_embed_2d = torch.cat([embed_x, embed_y], dim=-1).to(dtype)

        gap = feat_dim - pos_embed_2d.size(1)
        assert gap >= 0
        if gap > 0:
            assert ndim == 3
            padding = torch.zeros((pos_embed_2d.size(0), gap), dtype=dtype,
                                  device=voxel_coord_win.device)
            pos_embed_2d = torch.cat([pos_embed_2d, padding], dim=1)
        else:
            assert ndim == 2

        return pos_embed_2d
        #  [M', d]

    @torch.no_grad()
    def get_flat2batch_inds(self, voxel_coors_win, win_num_xy):
        """ get inds for inter region flat to batch wise
        Args:
            voxel_coors_win (Tensor): voxel coordinates for inter region
                attn, of shape (M', 1+3) where order is b, z, y, x
            win_num_xy: number of windows in x and y [win_num_x, win_num_y]
        Returns:
            (Tensor) indices for conversion from flat to batch of shape (M', )
        """
        max_win_x = win_num_xy[0]
        max_win_y = win_num_xy[1]
        max_win_per_batch = max_win_x * max_win_y

        flat2batch_inds = voxel_coors_win[:, 0] * max_win_per_batch + \
                          voxel_coors_win[:, 2] * max_win_x + \
                          voxel_coors_win[:, 3]

        return flat2batch_inds
        # (M', )

    def flat2batch(self, feats, voxel_coors_win, flat2batch_inds,
                   win_num_xy):
        """ Get batch wise voxel feats and voxel coors for Inter Region Attn
        Args:
            feats (Tensor): shape (M', d)
            voxel_coors_win (Tensor): shape (M', 1+3)
            flat2batch_inds (Tensor): shape (M', )
            win_num_xy (List): [win_num_x, win_num_y]

        Returns:
            feat_win_batch (Tensor): features in 3d of shape
                (batch_size, max_win_per_batch, feat_dim)
        """
        dtype = feats.dtype
        device = feats.device
        feat_dim = feats.shape[-1]

        batch_size = voxel_coors_win[:, 0].max().item() + 1
        max_win_per_batch = win_num_xy[0] * win_num_xy[1]

        feat_win_batch = torch.zeros((batch_size * max_win_per_batch,
                                      feat_dim), dtype=dtype,
                                     device=device)
        # (B*max_wins, d)
        feat_win_batch[flat2batch_inds] = feats
        feat_win_batch = feat_win_batch.reshape((batch_size,
                                                 max_win_per_batch, feat_dim))

        # (B, max_wins, d)
        return feat_win_batch

    def get_key_padding_mask_intereg(self, voxel_coord_win,
                                     flat2batch_inds,
                                     win_num_xy):
        """ get the key padding mask to mask the attention for non-token
        voxels in each batch
        Args:
            voxel_coord_win (Tensor): voxel cocordinates (M', 1+3)
            flat2batch_inds (Tensor): indices flat2batch (M', )
            win_num_xy (list): [win_num_x, win_num_y]
        Returns:

        """
        num_all_voxel = voxel_coord_win.shape[0]  # num of voxels = M'
        key_padding = torch.ones((num_all_voxel, 1)).to(
            voxel_coord_win.device).bool()  # (M', 1)

        # get key padding in 3d
        key_padding_win_batch = self.flat2batch(key_padding, voxel_coord_win,
                                                flat2batch_inds, win_num_xy)
        # (B, max_wins, 1)
        key_padding_win_batch = key_padding_win_batch.logical_not().squeeze(2)

        return key_padding_win_batch  # (B, max_wins)


class VoxelFeaturePropagation(nn.Module):
    """ Class to propagate voxel features from Inter Region Attention back
    to Intra Region Attention features
    Args:
        d_model (int): dimension of the model
        last_layer (bool): Whether last layer of Encoder, defaults to False
    """

    def __init__(self, d_model, region_shape=(20, 20),
                 sparse_shape=(512, 512, 1), ):
        super().__init__()

        self.d_model = d_model
        self.region_shape = region_shape
        self.sparse_shape = sparse_shape

        self.linear = nn.Linear(2 * d_model, d_model, bias=False)
        self.norm = nn.LayerNorm(d_model)
        self.activation = _get_activation_fn("relu")

    def forward(self,
                voxel_feat_win_batch,
                flat2batch_inds,
                voxel_coord_win,
                voxel_feat,
                win_inds_bzyx_interreg,
                ):
        """ Foward for Voxel Feature Propagation
        Args:
            voxel_feat_win_batch (Tensor): Voxel Inter region features of
                shape (B, nwin, d)
            flat2batch_inds (Tensor): flat2batch indices of shape (M', )
            voxel_coord_win (tensor): voxel coordinates of aggregates feats
                in windows of shape (M', 4)
            voxel_feat (Tensor): voxel feats from intra region attn of shape (M, d)
            win_inds_bzyx_interreg (Tensor): win indices of inter region
                feats of shape (M, 4)
        Returns:
            voxel_feats (Tensor): Fused intre and inter-region voxel feats
                of shape (M, d)
        """

        # convert batch-wise feats to flat feats
        voxel_feat_win = self.batch2flat(voxel_feat_win_batch, flat2batch_inds)
        # (M', d)

        # get the inter-region feats propagated to intra-region voxel feats
        voxel_feat_interreg_propagate = self.map_voxel_center_to_point(
            win_inds_bzyx_interreg, voxel_feat_win, voxel_coord_win)
        # (M, d)

        # concatenate intra-region and inter-region voxel feats
        voxel_feats_intrainter = torch.cat([voxel_feat,
                                            voxel_feat_interreg_propagate],
                                           dim=1)
        # (M, 2d)

        # fuse intre and inter-region vox feats
        voxel_feats = self.linear(voxel_feats_intrainter)  # (M, d)
        voxel_feats = self.norm(voxel_feats)
        voxel_feats = self.activation(voxel_feats)

        return voxel_feats  # (M, d)

    def map_voxel_center_to_point(self, pts_coors, voxel_mean, voxel_coors):
        """Map voxel features to its corresponding points.

        Args:
            pts_coors (torch.Tensor): Voxel coordinate of each point.
            voxel_mean (torch.Tensor): Voxel features to be mapped.
            voxel_coors (torch.Tensor): Coordinates of valid voxels

        Returns:
            torch.Tensor: Features or centers of each point.
        """
        # Step 1: scatter voxel into canvas
        # Calculate necessary things for canvas creation
        canvas_z = round(self.sparse_shape[-1] / 1)
        canvas_y = round(self.sparse_shape[1] / self.region_shape[1])
        canvas_x = round(self.sparse_shape[0] / self.region_shape[0])

        # canvas_channel = voxel_mean.size(1)
        batch_size = int(pts_coors[:, 0].max().item()) + 1
        canvas_len = canvas_z * canvas_y * canvas_x * batch_size
        # Create the canvas for this sample
        canvas = voxel_mean.new_zeros(canvas_len, dtype=torch.long)
        # Only include non-empty pillars
        indices = (
                voxel_coors[:, 0] * canvas_z * canvas_y * canvas_x +
                voxel_coors[:, 1] * canvas_y * canvas_x +
                voxel_coors[:, 2] * canvas_x + voxel_coors[:, 3])
        # Scatter the blob back to the canvas
        canvas[indices.long()] = torch.arange(
            start=0, end=voxel_mean.size(0), device=voxel_mean.device)

        # Step 2: get voxel mean for each point
        voxel_index = (
                pts_coors[:, 0] * canvas_z * canvas_y * canvas_x +
                pts_coors[:, 1] * canvas_y * canvas_x +
                pts_coors[:, 2] * canvas_x + pts_coors[:, 3])
        assert voxel_index.max() <= canvas_len - 1, f"Voxel Index " \
                                                    f"{voxel_index.max()} is " \
                                                    f"more than " \
                                                    f"{canvas_len - 1}; batch " \
                                                    f"size is {batch_size}"
        voxel_inds = canvas[voxel_index.long()]
        center_per_point = voxel_mean[voxel_inds, ...]
        return center_per_point

    def batch2flat(self, voxel_feat_win_batch, flat2batch_inds):
        """ Convert batch-wise feats to flat
        Used in Voxel Feature propagation after Inter Region Attn
        Args:
            voxel_feat_win_batch (Tensor): Voxel Inter region features of
                shape (B, nwin, d)
            flat2batch_inds (Tensor): flat2batch indices of shape (M', )
        Returns:
            voxel_feat_win (Tensor): Voxel flat features of shape (M', d)
        """

        feat_dim = voxel_feat_win_batch.shape[-1]
        voxel_feat_win_batch = voxel_feat_win_batch.reshape(-1, feat_dim)
        # (B*n_win, d)
        voxel_feat_win = voxel_feat_win_batch[flat2batch_inds]  # (M', d)

        return voxel_feat_win  # (M', d)


class InterRegionTransformerEncoder(nn.Module):
    """ Inter Region Transformer Encoder with
    Voxel Feature Region Aggregation
    Inter Region Transformer
    Voxel Feature Propagation

    Args:
        d_model (int): Dimension of model
        nhead (int): number of attention heads
        dim_feedforward (int): FFN hidden dimension
        dropout (float): Dropout in attention
        activation (str): Name of activation "relu"
        layer_cfg (dict): layer configuration to choose type of attention
    """

    def __init__(self,
                 d_model,
                 deli_lay_cfg=None,
                 region_shape=(20, 20),
                 sparse_shape=(512, 512, 1),
                 ):
        super().__init__()
        self.voxel_feat_region_aggregation = VoxelFeatureRegionAggregation()
        self.inter_region_transformer = DeLiTrEncoderLayer(d_model,
                                                           deli_lay_cfg=deli_lay_cfg)
        self.voxel_feat_propagation = VoxelFeaturePropagation(d_model,
                                                              region_shape=region_shape,
                                                              sparse_shape=sparse_shape)

    def forward(self, voxel_feat,
                win_inds_bzyx_interreg,
                win_num_xy,
                ):
        """ Inter Region TF Encoder
        Args:
            voxel_feat (Tensor): Voxel feats of shape (M, d)
            win_inds_bzyx_interreg (Tensor): Window indices of each voxel
                of shape (M, 4) order: b, z, y, x
            win_num_xy (list): number of windows in x and y directions
        Returns:
            voxel_feats (Tensor): Fused intre and inter-region voxel feats
                of shape (M, d)
        """
        # Voxel feature aggregation
        voxel_feat_aggregation = self.voxel_feat_region_aggregation(
            voxel_feat,
            win_inds_bzyx_interreg,
            win_num_xy)

        voxel_feat_win_batch, pos_emb_win_batch, key_padding_win_batch, \
            flat2batch_inds, voxel_coord_win = voxel_feat_aggregation
        # (B, num_win_per_batch, d), (B, num_win_per_batch, d),
        # (B, num_win_per_batch), (M', ), (M', 4)

        # Inter Region Transformer
        voxel_feat_win_batch = voxel_feat_win_batch + pos_emb_win_batch
        voxel_feat_win_batch = voxel_feat_win_batch.permute(1, 0, 2)
        # (num_win_per_batch, B, d)
        voxel_feat_win_batch = self.inter_region_transformer(
            voxel_feat_win_batch, key_padding_win_batch)
        # (num_win_per_batch, B, d)
        voxel_feat_win_batch = voxel_feat_win_batch.permute(1, 0, 2)
        # (B, num_win_per_batch, d)

        # Voxel Feature Propagation to Intra region
        voxel_feats = self.voxel_feat_propagation(voxel_feat_win_batch,
                                                  flat2batch_inds,
                                                  voxel_coord_win,
                                                  voxel_feat,
                                                  win_inds_bzyx_interreg)
        # (M, d)

        return voxel_feats  # (M, d)


class DeLiTrEncoderLayer(nn.Module):
    """
    This is DeLiTr encoder layer used in both intra and inter-region attention
    """

    def __init__(self,
                 d_model,
                 deli_lay_cfg=None):
        super().__init__()

        self.embed_dim = d_model
        width_multiplier = deli_lay_cfg['width_multiplier']
        dextra_depth = deli_lay_cfg['dextra_depth']
        dextra_dropout = deli_lay_cfg['dextra_dropout']
        dextra_proj = deli_lay_cfg['dextra_proj']
        attn_dropout = deli_lay_cfg['attn_dropout']
        dropout = deli_lay_cfg['dropout']
        act_dropout = deli_lay_cfg['act_dropout']
        ffn_dropout = deli_lay_cfg['ffn_dropout']
        enc_ffn_red = deli_lay_cfg['enc_ffn_red']
        norm_type = deli_lay_cfg['norm_type']
        act_type = deli_lay_cfg['act_type']
        normalize_before = deli_lay_cfg['normalize_before']
        assert d_model % dextra_proj == 0

        self.proj_dim = d_model // dextra_proj
        max_groups = 2 ** math.ceil(math.log(d_model // 32, 2))
        self.dextra_layer = DExTraUnit(in_features=self.embed_dim,
                                       in_proj_features=self.proj_dim,
                                       out_features=self.proj_dim,
                                       width_multiplier=width_multiplier,
                                       dextra_depth=dextra_depth,
                                       dextra_dropout=dextra_dropout,
                                       max_glt_groups=max_groups,
                                       act_type='gelu',
                                       use_bias=True,
                                       norm_type='ln',
                                       glt_shuffle=True,
                                       )

        self.self_attn = SingleHeadAttention(q_in_dim=self.proj_dim,
                                             k_in_dim=self.proj_dim,
                                             v_in_dim=self.proj_dim,
                                             proj_dim=self.proj_dim,
                                             out_dim=self.embed_dim,
                                             dropout=attn_dropout,
                                             bias=True)

        self.self_attn_layer_norm = get_norm_layer(name=norm_type,
                                                   out_features=self.embed_dim)
        self.dropout = dropout
        self.norm_fn = norm_type
        self.act_type = act_type
        self.activation_fn = get_activation_layer(name=act_type)
        self.activation_dropout = act_dropout
        self.normalize_before = normalize_before

        # Light-weight FFN
        self.ffn_dropout = ffn_dropout
        ffn_red_factor = enc_ffn_red
        assert self.embed_dim % ffn_red_factor == 0, '{}/{} should be a perfect divisor'.format(
            self.embed_dim,
            ffn_red_factor)
        light_ffn_dim = self.embed_dim // ffn_red_factor
        self.fc1 = get_weight_layer(name='linear',
                                    in_features=self.embed_dim,
                                    out_features=light_ffn_dim,
                                    use_bias=True)
        self.fc2 = get_weight_layer(name='linear',
                                    in_features=light_ffn_dim,
                                    out_features=self.embed_dim,
                                    use_bias=True)

        self.final_layer_norm = get_norm_layer(name=norm_type,
                                               out_features=self.embed_dim)

    def forward(self,
                x,
                encoder_padding_mask,
                attn_mask=None):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor): binary ByteTensor of shape
                `(batch, src_len)` where padding elements are indicated by ``1``.
            attn_mask (ByteTensor): binary tensor of shape (T_tgt, T_src), where
            T_tgt is the length of query, while T_src is the length of key,
            though here both query and key is x here,
            attn_mask[t_tgt, t_src] = 1 means when calculating embedding
            for t_tgt, t_src is excluded (or masked out), =0 means it is
            included in attention

        Returns:
            encoded output of shape `(seq_len, batch, embed_dim)`
        """
        residual = x  # (N, B, d)
        if self.normalize_before:
            x = self.self_attn_layer_norm(x)
        if attn_mask is not None:
            attn_mask = attn_mask.masked_fill(attn_mask.to(torch.bool), -1e8)

        x = self.dextra_layer(x)  # (N, B, d0)

        x, _ = self.self_attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=encoder_padding_mask,
            attn_mask=attn_mask
        )  # (N, B, d)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x  # (N, B, d)

        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)

        # Light-weight FFN
        residual = x
        if self.normalize_before:
            x = self.final_layer_norm(x)  # (N, B, d)
        x = self.activation_fn(self.fc1(x))  # (N, B, d/4)
        x = F.dropout(x, p=float(self.activation_dropout),
                      training=self.training)
        x = self.fc2(x)  # (N, B, d)
        x = F.dropout(x, p=self.ffn_dropout, training=self.training)
        x = residual + x  # (N, B, d)
        if not self.normalize_before:
            x = self.final_layer_norm(x)
        return x  # (N, B, d)


class DeLiVoTrEncoderLayer(nn.Module):
    ''' DeLiVoTr Encoder Layer, consist of:
    Intra-Region Transformer Encoder and
    Inter-Region Transformer Encoder

    Args:
        d_model (int): Dimension of model
        nhead (int): number of attention heads
        dim_feedforward (int): FFN hidden dimension
        dropout (float): Dropout in attention
        activation (str): Name of activation "relu"
        last_layer (bool): Whether last encoder layer, defaults to False
        layer_cfg (dict): layer configuration to choose type of attention
    '''

    def __init__(self, d_model,
                 region_shape=(20, 20),
                 sparse_shape=(512, 512, 1),
                 deli_lay_cfg=None,
                 ):
        super().__init__()

        self.intra_region_tf_encoder = IntraRegionTransformerEncoder(
            d_model,
            deli_lay_cfg=deli_lay_cfg)

        # Inter region with windowed based inter-region
        self.inter_region_tf_encoder = InterRegionTransformerEncoder(
            d_model,
            deli_lay_cfg=deli_lay_cfg,
            region_shape=region_shape,
            sparse_shape=sparse_shape,
        )

    def forward(
            self,
            src,
            pos_dict,
            ind_dict,
            key_mask_dict,
            win_inds_bzyx_interreg,
            win_num_xy,
            using_checkpoint=False,
    ):
        """ Forward of VoxFormer Encoder Layer
        Args:
            src (Tensor): Voxel Features from VFE of shape (M, d)
            pos_dict (dict): pos embeddings intra-region of shape
                {0:(n_win, m_tok, C), 1:(n_win, m_tok, C),
                 2:(n_win, m_tok, C)}
            ind_dict (dict): flat2win indices of shape
                {0:(,), 1:(,), 2:(,)}
            key_mask_dict (dict): key padding mask intra-region
                of shape
                {0:(n_win, m_tok, C), 1:(n_win, m_tok, C),
                 2:(n_win, m_tok, C)}
            win_inds_bzyx_interreg (Tensor): Window indices of each voxel
                of shape (M, 4) order: b, z, y, x
            win_num_xy (list): number of windows in x and y directions
            using_checkpoint (bool): Whether to trade memory for
                computation, default False
        Returns:
            voxel_feats (Tensor): Fused intre and inter-region voxel feats
                of shape (M, d)
            or
            voxel_feat_win_batch, pos_emb_win_batch, key_padding_win_batch (
            Tensor) : voxel region feats of shape (B, num_win, d)
        """

        if using_checkpoint and self.training:
            intra_region_output = checkpoint(self.intra_region_tf_encoder,
                                             src,
                                             pos_dict,
                                             ind_dict,
                                             key_mask_dict)  # (M, d)
            # inter region with window region aggregation
            inter_region_output = checkpoint(self.inter_region_tf_encoder,
                                             intra_region_output,
                                             win_inds_bzyx_interreg,
                                             win_num_xy)  # (M, d)
        else:
            intra_region_output = self.intra_region_tf_encoder(src,
                                                               pos_dict,
                                                               ind_dict,
                                                               key_mask_dict)
            # (M, d)

            # Inter-region with window based
            inter_region_output = self.inter_region_tf_encoder(
                intra_region_output,
                win_inds_bzyx_interreg,
                win_num_xy)
            # (M, d)

        return inter_region_output  # (M, d)


def _get_activation_fn(activation: str):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")
