# Modified from SST Input Layer

import torch
from mmcv.runner import auto_fp16
from torch import nn

from mmdet3d.models.builder import MIDDLE_ENCODERS
from ...ops.delivotr_ops import (flat2window_v2, window2flat_v2,
                                 get_inner_win_inds,
                                 get_flat2win_inds_v2, get_window_coors,
                                 get_win_inds_bzyx)


@MIDDLE_ENCODERS.register_module()
class DeLiVoTrInputLayer(nn.Module):
    """
    DeLiVoTr Encoder Input Layer
    modified from SST Input Layer

    There are 3 things to be done in this class:
    1. Regional Grouping : assign window indices to each voxel.
    2. Voxel drop and region batching: see SST paper for details
    3. Pre-computing the transformation information for converting
        flat features ([M x d]) to region features ([R, T, d]).
        R is the number of regions containing at most T tokens (voxels).
        See function flat2window and window2flat for details.

    Args::
        drop_info (dict): drop configuration for region batching.
        region_shape (tuple[int]): (num_x, num_y).
            Each region is divided to num_x * num_y pillars
            (including empty pillars).
        sparse_shape (tuple[int]): Sparse shape of full range of point cloud
            in voxels. Eg: (512, 512, 1)
        shuffle_voxels (bool): Shuffle the voxels, defaults to True
        debug (bool): To debug the result with assertion. defaults to False
        normalize_pos (bool): Normalize for position embedding,
            defaults to False
        pos_temperature (int): Used for pos embedding, defaults to 10000
    """

    def __init__(self,
                 drop_info,
                 region_shape,
                 sparse_shape,
                 shuffle_voxels=True,
                 debug=False,
                 normalize_pos=False,
                 pos_temperature=10000,
                 mute=False,
                 ):
        super().__init__()
        self.fp16_enabled = False
        self.meta_drop_info = drop_info
        self.region_shape = region_shape
        self.sparse_shape = sparse_shape
        self.shuffle_voxels = shuffle_voxels
        self.debug = debug
        self.normalize_pos = normalize_pos
        self.pos_temperature = pos_temperature
        self.mute = mute

    @auto_fp16(apply_to=('voxel_feat',))
    def forward(self, voxel_feats, voxel_coors, batch_size=None):
        """
        Args:
            voxel_feats (Tensor): Voxel features of shape (M, d)
                N is the voxel num in the batch.
            voxel_coors (Tensor): voxel coors of shape (M, 4), [b, z, y, x]
        Returns:
            voxel_info (dict): A dict containing information of voxels as
                follows:
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
        """

        self.set_drop_info()
        voxel_coors = voxel_coors.long()

        if self.shuffle_voxels:
            # shuffle the voxels to make the drop process uniform.
            shuffle_inds = torch.randperm(len(voxel_feats))
            voxel_feats = voxel_feats[shuffle_inds]
            voxel_coors = voxel_coors[shuffle_inds]

        voxel_info = self.window_partition(voxel_coors)
        voxel_info['voxel_feats'] = voxel_feats
        voxel_info['voxel_coors'] = voxel_coors
        voxel_info = self.drop_voxel(voxel_info,
                                     1)  # voxel_info is updated in this
        # function
        # WINDOWSHIFT

        voxel_feats = voxel_info['voxel_feats']  # after dropping
        voxel_coors = voxel_info['voxel_coors']

        for i in range(1):  # WINDOWSHIFT
            voxel_info[f'flat2win_inds_shift{i}'] = \
                get_flat2win_inds_v2(voxel_info[f'batch_win_inds_shift{i}'],
                                     voxel_info[f'voxel_drop_level_shift{i}'],
                                     self.drop_info, debug=True)
            # {0:(,), 1:(,), 2:(,)}

            voxel_info[f'pos_dict_shift{i}'] = \
                self.get_pos_embed(voxel_info[f'flat2win_inds_shift{i}'],
                                   voxel_info[f'coors_in_win_shift{i}'],
                                   voxel_feats.size(1), voxel_feats.dtype)
            #  {0:(n_win, m_tok, C), 1:(n_win, m_tok, C), 2:(n_win, m_tok, C)}

            voxel_info[f'key_mask_shift{i}'] = \
                self.get_key_padding_mask(
                    voxel_info[f'flat2win_inds_shift{i}'])
            #  {0:(n_win, m_tok, C), 1:(n_win, m_tok, C), 2:(n_win, m_tok, C)}

        # Inter Region Parameters
        # get the window x,y,z indices of each voxel
        voxel_info['win_inds_bzyx_interreg'] = get_win_inds_bzyx(
            voxel_coors, self.sparse_shape, self.region_shape)  # (N', 1+3)
        voxel_info['win_num_xy'] = (self.sparse_shape[0] //
                                    self.region_shape[0] + 1,
                                    self.sparse_shape[1] //
                                    self.region_shape[1] + 1)  # (26, 26)

        if self.debug:
            coors_3d_dict_shift0 = flat2window_v2(voxel_coors, voxel_info[
                'flat2win_inds_shift0'])
            coors_2d = window2flat_v2(coors_3d_dict_shift0,
                                      voxel_info['flat2win_inds_shift0'])
            assert (coors_2d == voxel_coors).all()

        if self.shuffle_voxels:
            voxel_info['shuffle_inds'] = shuffle_inds

        return voxel_info
        # {'batch_win_inds_shift0': (N',),
        #  'coors_in_win_shift0': (N', 3),
        #  'voxel_feats': (N', C),
        #  'voxel_coors': (N', 1+3),
        #  'voxel_keep_inds': (N',),
        #  'voxel_drop_level_shift0': (N',),
        #  'flat2win_inds_shift0': {0:(,), 1:(,), 2:(,)},
        #  'pos_dict_shift0': {0:(n_win, m_tok, C), 1:(n_win, m_tok, C),
        #  2:(n_win, m_tok, C)},
        #  'key_mask_shift0': {0:(n_win, m_tok, C), 1:(n_win, m_tok, C),
        #  2:(n_win, m_tok, C)},
        #  'shuffle_inds': (N,),
        #  'win_inds_bzyx_interreg': (N', 1+3),
        #  'win_num_xy': (win_x, win_y)}

    def drop_single_shift(self, batch_win_inds):
        drop_info = self.drop_info
        drop_lvl_per_voxel = -torch.ones_like(batch_win_inds)
        inner_win_inds = get_inner_win_inds(batch_win_inds)
        bincount = torch.bincount(batch_win_inds)
        num_per_voxel_before_drop = bincount[batch_win_inds]  #
        target_num_per_voxel = torch.zeros_like(batch_win_inds)

        for dl in drop_info:
            max_tokens = drop_info[dl]['max_tokens']
            lower, upper = drop_info[dl]['drop_range']
            range_mask = (num_per_voxel_before_drop >= lower) & (
                    num_per_voxel_before_drop < upper)
            target_num_per_voxel[range_mask] = max_tokens
            drop_lvl_per_voxel[range_mask] = dl

        if self.debug:
            assert (target_num_per_voxel > 0).all()
            assert (drop_lvl_per_voxel >= 0).all()

        keep_mask = inner_win_inds < target_num_per_voxel
        return keep_mask, drop_lvl_per_voxel

    def drop_voxel(self, voxel_info, num_shifts):
        """
        To make it clear and easy to follow, we do not use loop to process two
        shifts.
        """

        batch_win_inds_s0 = voxel_info['batch_win_inds_shift0']
        num_all_voxel = batch_win_inds_s0.shape[0]

        voxel_keep_inds = torch.arange(num_all_voxel,
                                       device=batch_win_inds_s0.device,
                                       dtype=torch.long)

        keep_mask_s0, drop_lvl_s0 = self.drop_single_shift(batch_win_inds_s0)
        # (B*N, ) (B*N, )
        if self.debug:
            assert (drop_lvl_s0 >= 0).all()

        drop_lvl_s0 = drop_lvl_s0[keep_mask_s0]
        voxel_keep_inds = voxel_keep_inds[keep_mask_s0]
        batch_win_inds_s0 = batch_win_inds_s0[keep_mask_s0]

        if num_shifts == 1:
            voxel_info['voxel_keep_inds'] = voxel_keep_inds
            voxel_info['voxel_drop_level_shift0'] = drop_lvl_s0
            voxel_info['batch_win_inds_shift0'] = batch_win_inds_s0
            voxel_num_before_drop = len(voxel_info['voxel_coors'])
            voxel_info['voxel_feats'] = voxel_info['voxel_feats'][
                voxel_keep_inds]
            voxel_info['voxel_coors'] = voxel_info['voxel_coors'][
                voxel_keep_inds]

            # Some other variables need to be dropped.
            for k, v in voxel_info.items():
                if isinstance(v, torch.Tensor) and len(
                        v) == voxel_num_before_drop:
                    voxel_info[k] = v[voxel_keep_inds]
            return voxel_info

        batch_win_inds_s1 = voxel_info['batch_win_inds_shift1']
        batch_win_inds_s1 = batch_win_inds_s1[keep_mask_s0]

        keep_mask_s1, drop_lvl_s1 = self.drop_single_shift(batch_win_inds_s1)
        if self.debug:
            assert (drop_lvl_s1 >= 0).all()

        # drop data in first shift again
        drop_lvl_s0 = drop_lvl_s0[keep_mask_s1]
        voxel_keep_inds = voxel_keep_inds[keep_mask_s1]
        batch_win_inds_s0 = batch_win_inds_s0[keep_mask_s1]

        drop_lvl_s1 = drop_lvl_s1[keep_mask_s1]
        batch_win_inds_s1 = batch_win_inds_s1[keep_mask_s1]

        voxel_info['voxel_keep_inds'] = voxel_keep_inds
        voxel_info['voxel_drop_level_shift0'] = drop_lvl_s0
        voxel_info['batch_win_inds_shift0'] = batch_win_inds_s0
        voxel_info['voxel_drop_level_shift1'] = drop_lvl_s1
        voxel_info['batch_win_inds_shift1'] = batch_win_inds_s1
        voxel_keep_inds = voxel_info['voxel_keep_inds']

        voxel_num_before_drop = len(voxel_info['voxel_coors'])
        voxel_info['voxel_feats'] = voxel_info['voxel_feats'][voxel_keep_inds]
        voxel_info['voxel_coors'] = voxel_info['voxel_coors'][voxel_keep_inds]

        # Some other variables need to be dropped.
        for k, v in voxel_info.items():
            if isinstance(v, torch.Tensor) and len(v) == voxel_num_before_drop:
                voxel_info[k] = v[voxel_keep_inds]

        ### sanity check
        if self.debug and self.training:
            for dl in self.drop_info:
                max_tokens = self.drop_info[dl]['max_tokens']

                mask_s0 = drop_lvl_s0 == dl
                if not mask_s0.any():
                    if not self.mute:
                        print(
                            f'No voxel belongs to drop_level:{dl} in shift 0')
                    continue
                real_max = torch.bincount(batch_win_inds_s0[mask_s0]).max()
                assert real_max <= max_tokens, f'real_max({real_max}) > {max_tokens} in shift0'

                mask_s1 = drop_lvl_s1 == dl
                if not mask_s1.any():
                    if not self.mute:
                        print(
                            f'No voxel belongs to drop_level:{dl} in shift 1')
                    continue
                real_max = torch.bincount(batch_win_inds_s1[mask_s1]).max()
                assert real_max <= max_tokens, f'real_max({real_max}) > {max_tokens} in shift1'
        ###
        return voxel_info

    @torch.no_grad()
    def window_partition(self, coors):
        voxel_info = {}
        for i in range(1):  # WINDOWSHIFT
            batch_win_inds, coors_in_win = get_window_coors(coors,
                                                            self.sparse_shape,
                                                            self.region_shape,
                                                            i == 1)
            # # (B*N,), (B*N, 3)
            voxel_info[f'batch_win_inds_shift{i}'] = batch_win_inds
            voxel_info[f'coors_in_win_shift{i}'] = coors_in_win

        return voxel_info

    @torch.no_grad()
    def get_pos_embed(self, inds_dict, coors_in_win, feat_dim, dtype):
        """
        Args:
        coors_in_win: shape=[N, 3], order: z, y, x
        """

        # [N,]
        region_shape = self.region_shape
        if len(region_shape) == 2:
            ndim = 2
            win_x, win_y = region_shape
            win_z = 0
        elif region_shape[-1] == 1:
            ndim = 2
            win_x, win_y = region_shape[:2]
            win_z = 0
        else:
            win_x, win_y, win_z = region_shape
            ndim = 3

        assert coors_in_win.size(1) == 3
        z, y, x = coors_in_win[:, 0] - win_z / 2, coors_in_win[:,
                                                  1] - win_y / 2, coors_in_win[
                                                                  :,
                                                                  2] - win_x / 2
        assert (x >= -win_x / 2 - 1e-4).all()
        assert (x <= win_x / 2 - 1 + 1e-4).all()

        if self.normalize_pos:
            x = x / win_x * 2 * 3.1415  # [-pi, pi]
            y = y / win_y * 2 * 3.1415  # [-pi, pi]
            z = z / win_z * 2 * 3.1415  # [-pi, pi]

        pos_length = feat_dim // ndim
        # [pos_length]
        inv_freq = torch.arange(
            pos_length, dtype=torch.float32, device=coors_in_win.device)
        inv_freq = self.pos_temperature ** (2 * (inv_freq // 2) / pos_length)

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
                                  device=coors_in_win.device)
            pos_embed_2d = torch.cat([pos_embed_2d, padding], dim=1)
        else:
            assert ndim == 2

        pos_embed_dict = flat2window_v2(
            pos_embed_2d, inds_dict)

        return pos_embed_dict
        #  {0:(n_win, m_tok, C), 1:(n_win, m_tok, C), 2:(n_win, m_tok, C)}

    @torch.no_grad()
    def get_key_padding_mask(self, ind_dict):
        num_all_voxel = len(ind_dict['voxel_drop_level'])
        key_padding = torch.ones((num_all_voxel, 1)).to(
            ind_dict['voxel_drop_level'].device).bool()

        window_key_padding_dict = flat2window_v2(key_padding, ind_dict)

        # logical not. True means masked
        for key, value in window_key_padding_dict.items():
            window_key_padding_dict[key] = value.logical_not().squeeze(2)

        return window_key_padding_dict
        #  {0:(n_win, m_tok, C), 1:(n_win, m_tok, C), 2:(n_win, m_tok, C)}

    def set_drop_info(self):
        if hasattr(self, 'drop_info'):
            return
        meta = self.meta_drop_info
        if isinstance(meta, tuple):
            if self.training:
                self.drop_info = meta[0]
            else:
                self.drop_info = meta[1]
        else:
            self.drop_info = meta
        print(f'drop_info is set to {self.drop_info}, in input_layer')

