import torch
import torch.nn as nn
from mmcv.ops import dynamic_scatter


class DynamicScatterCustom(nn.Module):

    def __init__(self, voxel_size, point_cloud_range, average_points: bool):
        super(DynamicScatterCustom, self).__init__()
        """Scatters points into voxels, used in the voxel encoder with
           dynamic voxelization

        **Note**: The CPU and GPU implementation get the same output, but
        have numerical difference after summation and division (e.g., 5e-7).

        Args:
            average_points (bool): whether to use avg pooling to scatter
                points into voxel voxel_size (list): list [x, y, z] size
                of three dimension
            point_cloud_range (list):
                [x_min, y_min, z_min, x_max, y_max, z_max]
        """
        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range
        self.average_points = average_points

    def forward_single(self, points, coors):
        reduce = 'mean' if self.average_points else 'max'
        return dynamic_scatter(points.contiguous(), coors.contiguous(), reduce)

    def forward(self, points, coors):
        """
        Args:
            input: NC points
        """
        if coors.size(-1) == 3:
            return self.forward_single(points, coors)
        else:
            batch_size = coors[:, 0].max().item() + 1
            voxels, voxel_coors = [], []
            for i in range(batch_size):
                inds = torch.where(coors[:, 0] == i)
                voxel, voxel_coor = self.forward_single(
                    points[inds], coors[inds][:, 1:])
                coor_pad = nn.functional.pad(
                    voxel_coor, (1, 0), mode='constant', value=i)
                voxel_coors.append(coor_pad)
                voxels.append(voxel)
            features = torch.cat(voxels, dim=0)
            feature_coors = torch.cat(voxel_coors, dim=0)

            return features, feature_coors

    def __repr__(self):
        tmpstr = self.__class__.__name__ + '('
        tmpstr += 'voxel_size=' + str(self.voxel_size)
        tmpstr += ', point_cloud_range=' + str(self.point_cloud_range)
        tmpstr += ', average_points=' + str(self.average_points)
        tmpstr += ')'
        return tmpstr