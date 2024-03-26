from ..registry import DETECTORS
from .single_stage import SingleStageDetector

import torch.nn as nn
import torch
import torch.nn.functional as F
from det3d.models.utils.cswin import CSWinBlock
from det3d.models.utils.set_transformer import SetBlock

pc_range = [0.3, -3.14368, -2.0, 75.18, 3.14368, 4.0]
voxel_size = [0.065, 0.00307, 0.15]
scale_factor = 8
x_size = 144
y_size = 256
meshgrid = [[0, x_size - 1, x_size], [0, y_size - 1, y_size]]
batch_y, batch_x = torch.meshgrid(*[torch.linspace(it[0], it[1], int(it[2])) for it in meshgrid])
batch_x = batch_x + 0.5
batch_y = batch_y + 0.5
bev_pos = torch.cat([batch_x[None], batch_y[None]], dim=0)[None]
bev_pos = bev_pos.permute(0, 2, 3, 1)  # 1, W,
r = bev_pos[..., 1] * voxel_size[0] * scale_factor + pc_range[0]
phi = bev_pos[..., 0] * voxel_size[1] * scale_factor + pc_range[1]
x = r * torch.cos(phi)
y = r * torch.sin(phi)
bev_pos = torch.stack([x, y, r, phi], dim=3)


@DETECTORS.register_module
class VoxelNet(SingleStageDetector):
    def __init__(
            self,
            reader,
            backbone,
            neck,
            bbox_head,
            seg_head,
            part_head=None,
            train_cfg=None,
            test_cfg=None,
            pretrained=None,
    ):
        super(VoxelNet, self).__init__(
            reader, backbone, neck, bbox_head, seg_head, part_head, train_cfg, test_cfg, pretrained
        )
        self.times = []

    def extract_feat_hard(self, data):

        input_features = self.reader(data["features"], data["num_voxels"])

        x, voxel_feature = self.backbone(
            input_features, data["coors"], data["batch_size"], data["input_shape"]
        )

        if self.with_neck:
            x = self.neck(x)

        return x, voxel_feature

    def extract_feat_dynamic(self, data):
        input_features, unq = self.reader(data)
        x, voxel_feature = self.backbone(
            input_features, unq, data["batch_size"], data["grid_size"]
        )

        if self.with_neck:
            x = self.neck(x)

        return x, voxel_feature

    def forward(self, example, return_loss=True, **kwargs):
        preds = {}
        # hard voxelization
        if 'voxels' in example:
            voxels = example["voxels"]
            coordinates = example["coordinates"]
            num_points_in_voxel = example["num_points"]
            num_voxels = example["num_voxels"]

            batch_size = len(num_voxels)

            data = dict(
                features=voxels,
                num_voxels=num_points_in_voxel,
                coors=coordinates,
                batch_size=batch_size,
                input_shape=example["shape"][0],
                prev_context=kwargs.get('prev_context', [])
            )
            extract_feat = self.extract_feat_hard
        else:
            num_points_per_sample = example["num_points"]
            batch_size = len(num_points_per_sample)

            data = dict(
                points=example['points'],
                grid_ind=example['grid_ind'],
                num_points=num_points_per_sample,
                batch_size=batch_size,
                voxel_size=example['voxel_size'][0],
                pc_range=example['pc_range'][0],
                grid_size=example['grid_size'][0],
                prev_context=kwargs.get('prev_context', [])
            )
            extract_feat = self.extract_feat_dynamic

        if self.seg_head:
            x, voxel_feature = extract_feat(data)
            preds.update(self.seg_head(voxel_feature['conv1'].dense(), x))


        else:
            x, _ = extract_feat(data)
        if self.bbox_head:
            preds.update(self.bbox_head(x))
        if return_loss:
            loss = {}
            if self.bbox_head:
                loss.update(self.bbox_head.loss(example, preds))
            if self.seg_head:
                loss.update(self.seg_head.loss(example, preds))
            return loss
        else:
            ret_dict = {}

            if self.bbox_head:
                ret_dict['det'] = self.bbox_head.predict(example, preds, self.test_cfg)
            if self.seg_head:
                ret_dict['seg'] = self.seg_head.predict(example, preds, self.test_cfg)

            return ret_dict

    def forward_two_stage(self, example, return_loss=True, **kwargs):
        voxels = example["voxels"]
        coordinates = example["coordinates"]
        num_points_in_voxel = example["num_points"]
        num_voxels = example["num_voxels"]

        batch_size = len(num_voxels)

        data = dict(
            features=voxels,
            num_voxels=num_points_in_voxel,
            coors=coordinates,
            batch_size=batch_size,
            input_shape=example["shape"][0],
        )

        x, voxel_feature = self.extract_feat(data)

        bev_feature = x
        preds = self.bbox_head(x)

        # manual deepcopy ...
        new_preds = []
        for pred in preds:
            new_pred = {}
            for k, v in pred.items():
                new_pred[k] = v.detach()

            new_preds.append(new_pred)

        boxes = self.bbox_head.predict(example, new_preds, self.test_cfg)

        if return_loss:
            return boxes, bev_feature, voxel_feature, self.bbox_head.loss(example, preds)
        else:
            return boxes, bev_feature, voxel_feature, None


@DETECTORS.register_module
class VoxelNetV3(SingleStageDetector):
    def __init__(
            self,
            reader,
            backbone,
            neck,
            bbox_head,
            seg_head,
            part_head=None,
            train_cfg=None,
            test_cfg=None,
            pretrained=None,
    ):
        super(VoxelNetV3, self).__init__(
            reader, backbone, neck, bbox_head, seg_head, part_head, train_cfg, test_cfg, pretrained
        )
        self.times = []

        self.bev_pos = bev_pos

        depth = 2
        self.attns = nn.ModuleList([
            SetBlock(
                in_dim=256, embed_dim_scale=1, num_heads=4, reso=(144, 256), mlp_ratio=4.,
                qkv_bias=True, qk_scale=None, H_sp=144, W_sp=1, H=4, W=8,
                drop=0.1, attn_drop=0.1,
                drop_path=0.1, norm_layer=nn.LayerNorm, pos=self.bev_pos, shift=False if i%2 == 0 else True)
            for i in range(depth)])


    def extract_feat_hard(self, data):

        input_features = self.reader(data["features"], data["num_voxels"])

        x, voxel_feature = self.backbone(
            input_features, data["coors"], data["batch_size"], data["input_shape"]
        )

        #### CSWin ####
        x = x.permute(0, 1, 3, 2).contiguous()
        B, C, H, W = x.shape
        x = x.view(B, C, -1)
        x = x.permute(0, 2, 1)

        for attn in self.attns:
            x = attn(x)

        x = x.permute(0, 2, 1)
        x = x.view(B, C, H, W)
        x = x.permute(0, 1, 3, 2).contiguous()

        if self.with_neck:
            x = self.neck(x)

        return x, voxel_feature

    def extract_feat_dynamic(self, data):
        input_features, unq = self.reader(data)
        x, voxel_feature = self.backbone(
            input_features, unq, data["batch_size"], data["grid_size"]
        )

        if self.with_neck:
            x = self.neck(x)

        return x, voxel_feature

    def forward(self, example, return_loss=True, **kwargs):
        preds = {}
        # hard voxelization
        if 'voxels' in example:
            voxels = example["voxels"]
            coordinates = example["coordinates"]
            num_points_in_voxel = example["num_points"]
            num_voxels = example["num_voxels"]

            batch_size = len(num_voxels)

            frame_ids = [metadata['token'] for metadata in example['metadata']]

            data = dict(
                features=voxels,
                num_voxels=num_points_in_voxel,
                coors=coordinates,
                batch_size=batch_size,
                input_shape=example["shape"][0],
                prev_context=kwargs.get('prev_context', []),
                frame_ids=frame_ids,
            )
            extract_feat = self.extract_feat_hard
        else:
            num_points_per_sample = example["num_points"]
            batch_size = len(num_points_per_sample)

            data = dict(
                points=example['points'],
                grid_ind=example['grid_ind'],
                num_points=num_points_per_sample,
                batch_size=batch_size,
                voxel_size=example['voxel_size'][0],
                pc_range=example['pc_range'][0],
                grid_size=example['grid_size'][0],
                prev_context=kwargs.get('prev_context', [])
            )
            extract_feat = self.extract_feat_dynamic

        if self.seg_head:
            x, voxel_feature = extract_feat(data)
            preds.update(self.seg_head(voxel_feature['conv1'].dense(), x))

        else:
            x, _ = extract_feat(data)

        if self.bbox_head:
            preds.update(self.bbox_head(x))
        if return_loss:
            loss = {}
            if self.bbox_head:
                loss.update(self.bbox_head.loss(example, preds))
            if self.seg_head:
                loss.update(self.seg_head.loss(example, preds))
            return loss
        else:
            ret_dict = {}

            if self.bbox_head:
                ret_dict['det'] = self.bbox_head.predict(example, preds, self.test_cfg)
            if self.seg_head:
                ret_dict['seg'] = self.seg_head.predict(example, preds, self.test_cfg)

            return ret_dict