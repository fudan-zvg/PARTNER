import itertools
import logging

from det3d.utils.config_tool import get_downsample_factor
nsweeps = 1
super_tasks = ['det']
# use relative velocity and orientation
rectify = False

voxel_generator = dict(
    # range=[0.3, -3.158, -2.0, 75.5, 3.158, 4.0],
    # voxel_size=[0.1, 0.0042, 0.15], # 1504*1504
    range=[0.3, -3.14368, -2.0, 75.18, 3.14368, 4.0],
    voxel_size=[0.065, 0.00307, 0.15],  # 1152, 2048
    max_points_in_voxel=5,
    max_voxel_num=150000,
    voxel_shape='cylinder',
    return_density=False,
    dynamic=False,
    nsectors=1,
)
tasks = [
    dict(num_class=1, class_names=['Vehicle',])
]
class_names = list(itertools.chain(*[t["class_names"] for t in tasks]))

# training and testing settings
target_assigner = dict(
    tasks=tasks,
)

max_volumn_space = [ 75.18, 3.14368, 4.0 ]
min_volumn_space = [0.3, -3.14368, -2.0]
grid_size = [1152, 2048, 40]


# model settings
bbox_head = None
seg_head = None
part_head = None
if 'det' in super_tasks:
    bbox_head = dict(
        type="E2ESWVoteHead",
        in_channels=sum([256, 256]),
        tasks=tasks,
        dataset='waymo',
        weight=2,
        code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        common_heads={'reg': (2, 2), 'height': (1, 2), 'dim': (3, 2), 'rot': (2, 2)},  # (output_channel, num_conv)
        voxel_shape=voxel_generator['voxel_shape'],
        voxel_generator=voxel_generator,
        out_size_factor=8,
        SET_CRIT_CONFIG={'weight_dict': {'loss_ce': 1, 'loss_bbox': 2, 'loss_vote': 0.25, 'loss_vote_cls': 1, 'loss_iou': 2},
                         'losses': [ 'loss_ce', 'loss_bbox', 'loss_vote', 'loss_vote_cls', 'loss_iou' ],
                         'sigma': 3.0,
                         'code_weights': [ 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 ],
                         'use_focal_loss': True,
                         'gamma': 2.0,
                         'alpha': 0.25
        },

        CODER_CONFIG={
            'code_size': 7,
            'encode_angle_by_sincos': True
        },

        MATCHER_CONFIG={'weight_dict': {'loss_ce': 0.25, 'loss_bbox': 0.75},
                        'losses': [ 'loss_ce', 'loss_bbox'],
                        'code_weights': [ 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 ],
                        'use_focal_loss': True,
                        'box_pred_metric': 'loss_bbox',
                        'use_heatmap': False
        },

        USE_FOCAL_LOSS=True,
        GT_PROCESSOR_CONFIG={'tasks': tasks,
                             'generate_votemap': True,
                             'max_volumn_space': max_volumn_space,
                             'min_volumn_space': min_volumn_space,
                             'grid_size': grid_size,
                             'feature_map_stride': 8,
                             'gaussian_overlap': 0.1,
                             'min_radius': 4,
                             'num_max_objs': 500,
                             'scale_factor': 2,
                             'mapping': {"Vehicle": 1}
                             },
        HEAD_CONFIG={'kernel_size': 3,
                     'sw_head_version': 'votev4',
                     'cls_head_version': 'v2',
                     'window_size': 7,
                     'sl_depth': [2],
                     'code_size': 7,
                     'encode_angle_by_sincos': True,
                     'iou_loss': True,
                     'iou_factor': 1,
                     'init_bias': -2.19,
                     'num_classes': tasks[0]['num_class']})

model = dict(
    type="VoxelNetV3",
    pretrained=None,
    reader=dict(
        type="VoxelFeatureExtractorV3",
        num_input_features=7,
    ),
    backbone=dict(
        type="SpMiddleResNetFHD", num_input_features=7, ds_factor=8),
    neck=dict(
        type="RPN",
        layer_nums=[5, 5],
        ds_layer_strides=[1, 2],
        ds_num_filters=[128, 256],
        us_layer_strides=[1, 2],
        us_num_filters=[256, 256],
        num_input_features=256,
        logger=logging.getLogger("RPN"),
        set_depth=2,
        set_h=4,
        set_w=8,
        set_drop=0.,
        set_attn_drop=0.,
        set_drop_path=0.1
    ),
    bbox_head=bbox_head,
    seg_head=seg_head,
    part_head=part_head)

assigner = dict(
    target_assigner=target_assigner,
    out_size_factor=get_downsample_factor(model),
    dense_reg=1,
    gaussian_overlap=0.1,
    max_objs=500,
    min_radius=2,
    voxel_shape=voxel_generator['voxel_shape'],
)


train_cfg = dict(assigner=assigner)

test_cfg = dict(
    post_center_limit_range=[-80, -80, -10.0, 80, 80, 10.0],
    nms=dict(
        nms_pre_max_size=4096,
        nms_post_max_size=500,
        nms_iou_threshold=0.7,
    ),
    score_threshold=0.1,
    pc_range=voxel_generator['range'],
    out_size_factor=get_downsample_factor(model),
    voxel_size=voxel_generator['voxel_size'],
    rectify=rectify,
)

# dataset settings
# dataset settings
dataset_type = "WaymoDataset"
data_root = "/cache/waymo/waymo_processed_data_v0_5_0"

db_sampler = dict(
    type="GT-AUG",
    enable=True,
    db_info_path="/cache/waymo/dbinfos_train_%dsweeps_withvelo.pkl" %nsweeps,
    sample_groups=[
        dict(Vehicle=15),
    ],
    db_prep_steps=[
        dict(
            filter_by_min_num_points=dict(
                Vehicle=5,
            )
        ),
        dict(filter_by_difficulty=[-1],),
    ],
    global_random_rotation_range_per_object=[0, 0],
    rate=1.0,
)


train_preprocessor = dict(
    mode="train",
    shuffle_points=True,
    global_rot_noise=[-0.78539816, 0.78539816],
    global_scale_noise=[0.95, 1.05],
    db_sampler=db_sampler,
    class_names=class_names,
    voxel_shape=voxel_generator['voxel_shape'],
)

val_preprocessor = dict(
    mode="val",
    shuffle_points=False,
    voxel_shape=voxel_generator['voxel_shape'],
    class_names=class_names,
)

train_pipeline = [
    dict(type="LoadPointCloudFromFile", dataset=dataset_type, super_tasks=super_tasks),
    dict(type="LoadPointCloudAnnotations", with_bbox=True),
    dict(type="Preprocess", cfg=train_preprocessor, super_tasks=super_tasks),
    dict(type="Voxelization", cfg=voxel_generator, super_tasks=super_tasks),
    dict(type="AssignLabel", cfg=train_cfg["assigner"],super_tasks=super_tasks,rectify=rectify),
    dict(type="Reformat", super_tasks=super_tasks),
]
test_pipeline = [
    dict(type="LoadPointCloudFromFile", dataset=dataset_type, super_tasks=super_tasks),
    dict(type="LoadPointCloudAnnotations", with_bbox=True),
    dict(type="Preprocess", cfg=val_preprocessor, super_tasks=super_tasks),
    dict(type="Voxelization", cfg=voxel_generator, super_tasks=super_tasks),
    dict(type="AssignLabel", cfg=train_cfg["assigner"],super_tasks=super_tasks,rectify=rectify),
    dict(type="Reformat", super_tasks=super_tasks),
]

train_anno = "/cache/waymo/waymo_processed_data_v0_5_0_infos_train_v2.pkl"
val_anno = "/cache/waymo/waymo_processed_data_v0_5_0_infos_val_v2.pkl"
test_anno = "/cache/waymo/waymo_processed_data_v0_5_0_infos_test_v2.pkl"

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=8,
    train=dict(
        type=dataset_type,
        root_path=data_root,
        info_path=train_anno,
        ann_file=train_anno,
        nsweeps=nsweeps,
        class_names=class_names,
        pipeline=train_pipeline,
        super_tasks=super_tasks,
        load_interval=1,
    ),
    val=dict(
        type=dataset_type,
        root_path=data_root,
        info_path=val_anno,
        mode=val_preprocessor['mode'],
        ann_file=val_anno,
        nsweeps=nsweeps,
        class_names=class_names,
        pipeline=test_pipeline,
        super_tasks=super_tasks,
        load_interval=1,
    ),
    test=dict(
        type=dataset_type,
        root_path=data_root,
        info_path=test_anno,
        ann_file=test_anno,
        nsweeps=nsweeps,
        class_names=class_names,
        pipeline=test_pipeline,
        super_tasks=super_tasks,
        mode='test',
        version='v1.0-test',
    ),
)


optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# optimizer
optimizer = dict(
    type="adam", amsgrad=0.0, wd=0.01, fixed_wd=True, moving_average=False,
)
lr_config = dict(
    type="one_cycle", lr_max=0.003, moms=[0.95, 0.85], div_factor=10.0, pct_start=0.4,
)

checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=5,
    hooks=[
        dict(type="TextLoggerHook"),
        # dict(type='TensorboardLoggerHook')
    ],
)
# yapf:enable
# runtime settings
total_epochs = 36
device_ids = range(8)
dist_params = dict(backend="nccl", init_method="env://")
log_level = "INFO"
work_dir = './work_dirs/{}/'.format(__file__[__file__.rfind('/') + 1:-3])
load_from = None
resume_from = None
workflow = [('train', 1)]