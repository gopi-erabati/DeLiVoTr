lib = True
lib_dir = 'lib'

voxel_size = (0.32, 0.32, 6)
region_shape = (12, 12, 1)  # 12 * 0.32m
point_cloud_range = [-74.88, -74.88, -2, 74.88, 74.88, 4]
sparse_shape = (468, 468, 1)
drop_info_training = {
    0: {'max_tokens': 30, 'drop_range': (0, 30)},
    1: {'max_tokens': 60, 'drop_range': (30, 60)},
    2: {'max_tokens': 100, 'drop_range': (60, 100000)},
}
drop_info_test = {
    0: {'max_tokens': 30, 'drop_range': (0, 30)},
    1: {'max_tokens': 60, 'drop_range': (30, 60)},
    2: {'max_tokens': 100, 'drop_range': (60, 100)},
    3: {'max_tokens': 144, 'drop_range': (100, 100000)},
}
drop_info = (drop_info_training, drop_info_test)

model = dict(
    type='DeLiVoTr',
    pts_voxel_layer=dict(
        voxel_size=voxel_size,
        max_num_points=-1,
        point_cloud_range=point_cloud_range,
        max_voxels=(-1, -1)
    ),
    pts_voxel_encoder=dict(
        type='DynamicVFECustom',
        in_channels=5,
        feat_channels=[64, 128],
        with_distance=False,
        voxel_size=voxel_size,
        with_cluster_center=True,
        with_voxel_center=True,
        point_cloud_range=point_cloud_range,
        norm_cfg=dict(type='naiveSyncBN1dCustom', eps=1e-3, momentum=0.01),
        with_centroid_aware_vox=True,
        centroid_to_point_pos_emb_dims=32,
    ),
    pts_middle_encoder=dict(
        type='DeLiVoTrInputLayer',
        drop_info=drop_info,
        region_shape=region_shape,
        sparse_shape=sparse_shape,
        shuffle_voxels=True,
        debug=True,
        normalize_pos=False,
        pos_temperature=10000,
    ),
    pts_backbone=dict(
        type='DeLiVoTrEncoder',
        d_model=128,
        enc_num_layers=6,
        checkpoint_layers=[],
        region_shape=region_shape,
        sparse_shape=sparse_shape,
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
            enc_ffn_red=2,
            norm_type='ln',
            act_type='relu',
            normalize_before=True,
        ),
        num_attached_conv=4,
        conv_kwargs=[
            dict(kernel_size=3, dilation=1, padding=1, stride=1),
            dict(kernel_size=3, dilation=1, padding=1, stride=1),
            dict(kernel_size=3, dilation=1, padding=1, stride=1),
            dict(kernel_size=3, dilation=2, padding=2, stride=1),
        ],
        conv_in_channel=128,
        conv_out_channel=128,
        conv_shortcut=True,
    ),
    pts_neck=dict(
        type='SECONDFPN',
        norm_cfg=dict(type='naiveSyncBN2d', eps=1e-3, momentum=0.01),
        in_channels=[128, ],
        upsample_strides=[1, ],
        out_channels=[128, ]
    ),
    bbox_head=dict(
        type='CenterHead',
        in_channels=128,
        tasks=[
            dict(num_class=3, class_names=['car', 'pedestrian', 'cyclist']),
        ],
        common_heads=dict(
            reg=(2, 2), height=(1, 2), dim=(3, 2), rot=(2, 2),
        ),
        share_conv_channel=64,
        bbox_coder=dict(
            type='CenterPointBBoxCoder',
            post_center_range=[-74.88, -74.88, -10.0, 74.88, 74.88, 10.0],
            max_num=4096,
            score_threshold=0.1,
            out_size_factor=1,
            voxel_size=voxel_size[:2],
            pc_range=point_cloud_range[:2],
            code_size=7),
        separate_head=dict(
            type='DCNSeparateHead', init_bias=-2.19, final_kernel=3,
            dcn_config=dict(
                type='DCN',
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                padding=1,
                groups=4,
                bias=False
            ),  # mmcv 1.2.6 doesn't support bias=True anymore
            norm_cfg=dict(type='naiveSyncBN2d', eps=1e-3, momentum=0.01),
        ),
        norm_cfg=dict(type='naiveSyncBN2d', eps=1e-3, momentum=0.01),
        loss_cls=dict(type='GaussianFocalLoss', reduction='mean'),
        loss_bbox=dict(type='L1Loss', reduction='mean', loss_weight=2),
        norm_bbox=True
    ),
    # model training and testing settings
    train_cfg=dict(
        grid_size=[468, 468, 1],
        voxel_size=voxel_size,
        out_size_factor=1,
        dense_reg=1,  # not used
        gaussian_overlap=0.1,
        max_objs=500,
        min_radius=2,
        point_cloud_range=point_cloud_range,
        code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    ),
    test_cfg=dict(
        post_center_limit_range=[-80, -80, -10, 80, 80, 10],
        max_per_img=500,
        max_pool_nms=False,
        min_radius=[4, 12, 10, 1, 0.85, 0.175],
        # not used in normal nms, task-wise
        score_threshold=0.1,
        pc_range=point_cloud_range[:2],  # seems not used
        out_size_factor=1,
        voxel_size=voxel_size[:2],
        nms_type='rotate',
        pre_max_size=4096,
        post_max_size=500,
        nms_thr=0.7
    )
)

# dataset settings
# D5 in the config name means the whole dataset is divided into 5 folds
# We only use one fold for efficient experiments
dataset_type = 'CustomWaymoDataset'
data_root = './data/waymo/kitti_format/'
file_client_args = dict(backend='disk')
class_names = ['Car', 'Pedestrian', 'Cyclist']
point_cloud_range = [-74.88, -74.88, -1.999, 74.88, 74.88,
                     3.999]  # add a small margin in z-axis
input_modality = dict(use_lidar=True, use_camera=False)
db_sampler = dict(
    data_root=data_root,
    info_path=data_root + 'waymo_dbinfos_train.pkl',
    rate=1.0,
    prepare=dict(
        filter_by_difficulty=[-1],
        filter_by_min_points=dict(Car=5, Pedestrian=10, Cyclist=10)),
    classes=class_names,
    sample_groups=dict(Car=15, Pedestrian=10, Cyclist=10),
    points_loader=dict(
        type='LoadPointsFromFileTanhWaymo',
        coord_type='LIDAR',
        load_dim=6,
        tanh_dim=[3, 4],
        use_dim=[0, 1, 2, 3, 4],
        file_client_args=file_client_args))

train_pipeline = [
    dict(
        type='LoadPointsFromFileTanhWaymo',
        coord_type='LIDAR',
        load_dim=6,
        use_dim=5,
        tanh_dim=[3, 4],
        file_client_args=file_client_args),
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=True,
        with_label_3d=True,
        file_client_args=file_client_args),
    dict(type='ObjectSample', db_sampler=db_sampler),
    dict(
        type='RandomFlip3D',
        sync_2d=False,
        flip_ratio_bev_horizontal=0.5,
        flip_ratio_bev_vertical=0.5),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-0.78539816, 0.78539816],
        scale_ratio_range=[0.95, 1.05]),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='PointShuffle'),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='Collect3D', keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
]
test_pipeline = [
    dict(
        type='LoadPointsFromFileTanhWaymo',
        coord_type='LIDAR',
        load_dim=6,
        use_dim=5,
        tanh_dim=[3, 4],
        file_client_args=file_client_args),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='GlobalRotScaleTrans',
                rot_range=[0, 0],
                scale_ratio_range=[1., 1.],
                translation_std=[0, 0, 0]),
            dict(type='RandomFlip3D'),
            dict(
                type='PointsRangeFilter', point_cloud_range=point_cloud_range),
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names,
                with_label=False),
            dict(type='Collect3D', keys=['points'])
        ])
]
# construct a pipeline for data and gt loading in show function
# please keep its loading function consistent with test_pipeline (e.g. client)
eval_pipeline = [
    dict(
        type='LoadPointsFromFileTanhWaymo',
        coord_type='LIDAR',
        load_dim=6,
        use_dim=5,
        tanh_dim=[3, 4],
        file_client_args=file_client_args),
    dict(
        type='DefaultFormatBundle3D',
        class_names=class_names,
        with_label=False),
    dict(type='Collect3D', keys=['points'])
]

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=8,
    train=dict(
        type='RepeatDataset',
        times=1,
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            ann_file=data_root + 'waymo_infos_train.pkl',
            split='training',
            pipeline=train_pipeline,
            modality=input_modality,
            classes=class_names,
            test_mode=False,
            # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
            # and box_type_3d='Depth' in sunrgbd and scannet dataset.
            box_type_3d='LiDAR',
            load_interval=1)),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'waymo_infos_val.pkl',
        split='training',
        pipeline=test_pipeline,
        modality=input_modality,
        classes=class_names,
        test_mode=True,
        box_type_3d='LiDAR'),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'waymo_infos_val.pkl',
        split='training',
        pipeline=test_pipeline,
        modality=input_modality,
        classes=class_names,
        test_mode=True,
        box_type_3d='LiDAR'))

evaluation = dict(interval=24, pipeline=eval_pipeline)

lr = 1e-5
optimizer = dict(
    type='AdamW',
    lr=lr,
    betas=(0.9, 0.999),  # the momentum is change during training
    weight_decay=0.05,
    paramwise_cfg=dict(custom_keys={'norm': dict(decay_mult=0.)}),
)
optimizer_config = dict(grad_clip=dict(max_norm=10, norm_type=2))
lr_config = dict(
    policy='cyclic',
    target_ratio=(100, 1e-3),
    cyclic_times=1,
    step_ratio_up=0.1,
)
momentum_config = None
runner = dict(type='EpochBasedRunner', max_epochs=24)

checkpoint_config = dict(interval=1)
# yapf:disable push
# By default we use textlogger hook and tensorboard
# For more loggers see
# https://mmcv.readthedocs.io/en/latest/api.html#mmcv.runner.LoggerHook
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = None
load_from = None
resume_from = None
workflow = [('train', 1)]

find_unused_parameters = True
# disable opencv multithreading to avoid system being overloaded
opencv_num_threads = 0
# set multi-process start method as `fork` to speed up the training
mp_start_method = 'fork'
