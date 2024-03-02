import torch
import torch.nn.functional as F
from mmcv.runner import force_fp32
from mmdet3d.core import bbox3d2result
from mmdet3d.models.detectors.base import Base3DDetector
from mmdet3d.models.builder import (build_voxel_encoder,
                                    build_middle_encoder, build_backbone,
                                    build_neck, build_head, DETECTORS)
from mmdet3d.ops import Voxelization


@DETECTORS.register_module()
class DeLiVoTr(Base3DDetector):
    """
    Deep and Light-weight Voxel Transformer (DeLiVoTr) for 3D Object Detection
    """
    def __init__(self,
                 pts_voxel_layer=None,
                 pts_voxel_encoder=None,
                 pts_middle_encoder=None,
                 pts_backbone=None,
                 pts_neck=None,
                 bbox_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(DeLiVoTr, self).__init__(init_cfg)

        # POINTS FEATURES : Points Voxel Layer, Points Voxel Encoder,
        # Points DeLiVoTrInputLayer, Pts backbone (DeLiVoTrEncoder)
        if pts_voxel_layer:
            self.pts_voxel_layer = Voxelization(**pts_voxel_layer)
        if pts_voxel_encoder:
            self.pts_voxel_encoder = build_voxel_encoder(
                pts_voxel_encoder)
        if pts_middle_encoder:
            self.pts_middle_encoder = build_middle_encoder(
                pts_middle_encoder)
        if pts_backbone:
            self.pts_backbone = build_backbone(pts_backbone)
        if pts_neck is not None:
            self.pts_neck = build_neck(pts_neck)
        else:
            self.pts_neck = None

        # build head
        bbox_head.update(train_cfg=train_cfg)
        bbox_head.update(test_cfg=test_cfg)
        self.bbox_head = build_head(bbox_head)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    @force_fp32(apply_to=('points'))
    def forward(self, points, return_loss=True, **kwargs):
        """Calls either forward_train or forward_test depending on whether
        return_loss=True.

        Note this setting will change the expected inputs. When
        `return_loss=True`, img and img_metas are single-nested (i.e.
        torch.Tensor and list[dict]), and when `resturn_loss=False`, img and
        img_metas should be double nested (i.e.  list[torch.Tensor],
        list[list[dict]]), with the outer list indicating test time
        augmentations.
        """
        if return_loss:
            return self.forward_train(points, **kwargs)
        else:
            return self.forward_test(points, **kwargs)

    def forward_train(self,
                      points,
                      gt_bboxes_3d,
                      gt_labels_3d,
                      img_metas,
                      gt_bboxes_3d_ignore=None):
        """
        Args:
            points (List[Tensor]): List of points tensor of shape (N, 5)
            gt_bboxes_3d (List[Tensor]): A list of tensors of batch length,
                each containing the ground truth 3D boxes of shape (num_box, 7)
            gt_labels_3d (List[Tensor]): A list of tensors of batch length,
                each containing the ground truth 3D boxes labels
                of shape (num_box, )
            img_metas (list[dict]): list indicates samples in a batch
            gt_bboxes_3d_ignore (None | list[Tensor]): Specify which
                bounding boxes can be ignored when computing the loss.

        Returns:
            dict [str, Tensor]: A dictionary of loss components
        """
        point_feats = self.extract_feat(points, img_metas)
        # returns BEV feat as list([B, C, H, W])

        outs = self.bbox_head(point_feats)
        loss_inputs = [gt_bboxes_3d, gt_labels_3d, outs]
        losses = self.bbox_head.loss(*loss_inputs)
        return losses

    def extract_feat(self, points, img_metas=None):
        """Extract Point Features"""

        # Point Features
        point_feats = self.extract_point_features(points)
        # returns BEV feat as list([B, C, H, W])

        return point_feats
        # list([B, C, H, W])

    @torch.no_grad()
    @force_fp32()
    def voxelize(self, points):
        """Apply dynamic voxelization to points.

        Args:
            points (list[torch.Tensor]): Points of each sample.

        Returns:
            tuple[torch.Tensor]: Concatenated points and coordinates.
        """
        coors = []
        # dynamic voxelization only provide a coors mapping
        for res in points:
            res_coors = self.pts_voxel_layer(res)  # (N, 3)
            coors.append(res_coors)
        points = torch.cat(points, dim=0)  # (B*N, 4)
        coors_batch = []
        for i, coor in enumerate(coors):
            coor_pad = F.pad(coor, (1, 0), mode='constant', value=i)  # (N,1+3)
            coors_batch.append(coor_pad)
        coors_batch = torch.cat(coors_batch, dim=0)  # (B*N, 1+3)
        return points, coors_batch  # (B*N, 4), (B*N, 1+3)

    def extract_point_features(self, points):
        """ Extract features of Points using encoder, middle encoder,
        backbone.
        Here points is list[Tensor] of batch """

        voxels, coors = self.voxelize(points)  # (B*N, 4), (B*N, 1+3)

        voxel_features, feature_coors = self.pts_voxel_encoder(voxels, coors)
        # (M, 128), (M, 1+3)
        # batch_size = coors[-1, 0].item() + 1

        x = self.pts_middle_encoder(voxel_features, feature_coors)
        # returns voxel_info as a dict with all info for next backbone

        x = self.pts_backbone(x)
        # it outputs BEV feat list (B, C, H, W)

        if self.pts_neck is not None:
            x = self.pts_neck(x)
        return x
        # it outputs BEV feat list (B, C, H, W)

    def forward_test(self,
                     points,
                     img_metas,
                     **kwargs):
        """
        Args:
            points (list[list[Tensor]]): List of points tensor of shape (N, 5)
            img_metas (list[list[dict]]): the outer list indicates test-time
                augs (multiscale, flip, etc.) and the inner list indicates
                images in a batch
        """
        for var, name in [(img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError('{} must be a list, but got {}'.format(
                    name, type(var)))

        num_augs = len(points)
        bbox_list = self.simple_test(points[0], img_metas[0], **kwargs)
        return bbox_list

    def simple_test(self, points, img_metas, rescale=False):
        """ Test function without test-time augmentation.

        Args:
            points (List[Tensor]): List of points tensor of shape (N, 5)
            img_metas (list[dict]): A list of image info where each dict
                has: 'img_Shape', 'flip' and other detailssee
                :class `mmdet3d.datasets.pipelines.Collect`.
        Returns:
            list[dict]: Predicted 3d boxes. Each list consists of a dict
            with keys: boxes_3d, scores_3d, labels_3d.
        """
        point_feats = self.extract_feat(points, img_metas)
        # returns BEV feat as list([B, C, H, W])

        outs = self.bbox_head(point_feats)
        bbox_list = self.bbox_head.get_bboxes(
            outs, img_metas, rescale=rescale)

        bbox_results = [
            bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels in bbox_list
        ]

        return bbox_results

    def aug_test(self, img, proj_img, proj_idxs, img_idxs, img_metas,
                 rescale=False):
        pass
