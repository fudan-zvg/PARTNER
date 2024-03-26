import logging
from collections import defaultdict
from det3d.core import box_np_ops
import torch
from det3d.torchie.cnn import kaiming_init
from torch import nn, double
from det3d.models.utils import Sequential
from ..registry import BBOX_HEADS
import copy
import numpy as np
from detectron2 import layers
from torch.nn import functional as F

from det3d.models.bbox_heads.e2e_modules import GroundTruthProcessor
from det3d.models.bbox_heads.center_head_parallel import CenterHeadSingle
from det3d.models.e2e_utils.box_coder_utils import CenterCoderV2, CenterCoder
from det3d.models.e2e_utils.matcher import TimeMatcher
from det3d.models.e2e_utils.set_crit import SetCriterion
from .swin_utils.sw2votev4_util import SwinTransformer as SwVoteHeadV4
    

@BBOX_HEADS.register_module
class E2ESWVoteHead(nn.Module):
    def __init__(self, 
                 in_channels=[128,],
                 tasks=[],
                 dataset='nuscenes',
                 weight=0.25,
                 code_weights=[],
                 common_heads=dict(),
                 logger=None,
                 init_bias=-2.19,
                 share_conv_channel=64,
                 num_hm_conv=2,
                 dcn_head=False,
                 voxel_shape='cuboid',
                 voxel_generator=None,
                 out_size_factor=4,
                 npixels=0,
                 SET_CRIT_CONFIG=dict(),
                 MATCHER_CONFIG=dict(),
                 USE_FOCAL_LOSS=True,
                 GT_PROCESSOR_CONFIG=dict(),
                 CODER_CONFIG=dict(),
                 HEAD_CONFIG=dict(),
    ):
        super(E2ESWVoteHead, self).__init__()
        head_conv = 64
        self.npixels = npixels
        self.period = 2 * np.pi
        self.dataset = dataset
        self.class_names = [t["class_name"] for t in tasks]
        self.voxel_shape = voxel_shape
        num_classes = [t["num_class"] for t in tasks]
        self.num_classes = num_classes

        ks = HEAD_CONFIG['kernal_size']
        self.sw_head_version = HEAD_CONFIG['sw_head_version', 'v1']
        self.window_size = HEAD_CONFIG['window_size', 7]
        self.sl_depths = HEAD_CONFIG['sl_depths', [2]]

        self.iou_loss = HEAD_CONFIG['iou_loss', False]
        self.iou_factor = HEAD_CONFIG['iou_factor', False]

        if self.sw_head_version == 'votev4':
            swhead = SwVoteHeadV4
        else:
            raise NotImplementedError
        
        out_ch_conv1 = in_channels // 2

        self.layer = swhead(mlp_ratio=1,embed_dim=out_ch_conv1,depths=self.sl_depths,window_size=self.window_size, in_ch=in_channels, use_patch_embed=True)

        cls_head_list = []
        for i in range(2):
            cls_head_list.append(nn.Sequential(
                nn.Conv2d(out_ch_conv1, out_ch_conv1, kernel_size=3, stride=1, padding=1, bias=True),
                nn.BatchNorm2d(out_ch_conv1),
                nn.ReLU()
            ))
            cls_head_list.append(
                nn.Conv2d(out_ch_conv1, HEAD_CONFIG['num_classes'], kernel_size=ks, stride=1, padding=ks//2)
            )
            self.cls_head = nn.Sequential(*cls_head_list)

        code_size = HEAD_CONFIG['code_size']
        if HEAD_CONFIG['encode_angle_by_sincos']:
            code_size += 1

        self.bbox_head = nn.Sequential(
            nn.Conv2d(out_ch_conv1, head_conv, kernel_size=ks, stride=1, padding=ks//2),
            nn.ReLU(inplace=True),
            nn.Conv2d(head_conv, code_size, kernel_size=ks, stride=1, padding=ks//2)
        )
        if self.iou_loss:
            self.iou_head = nn.Sequential(
                nn.Conv2d(out_ch_conv1, head_conv, kernel_size=ks, stride=1, padding=ks//2),
                nn.ReLU(inplace=True),
                nn.Conv2d(head_conv, 1, kernel_size=ks, stride=1, padding=ks//2)
            )

        self.vote_head = nn.Sequential(
            nn.Conv2d(in_channels, head_conv, kernel_size=ks, stride=1, padding=ks//2),
            nn.ReLU(inplace=True),
            nn.Conv2d(head_conv, 2, kernel_size=ks, stride=1, padding=ks//2)
        )

        self.vote_cls_head = nn.Sequential(
            nn.Conv2d(in_channels, out_ch_conv1, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch_conv1),
            nn.ReLU(),
            nn.Conv2d(out_ch_conv1, 1, kernel_size=ks, stride=1, padding=ks//2)
        )

        self._reset_parameters()

        self.cls_head[-1].bias.data.fill_(HEAD_CONFIG['init_bias'])

        box_coder_config = CODER_CONFIG
        box_coder_conifg['period'] = self.period
        box_coder = CenterCoder(**box_coder_config)

        set_crit_settings = SET_CRIT_CONFIG
        matcher_settings = MATCHER_CONFIG
        self.matcher_weights = matcher_settings['weights_dict']
        self.use_focal_loss = USE_FOCAL_LOSS
        self.box_coder = box_coder

        matcher_settings['box_coder'] = box_coder
        matcher_settings['period'] = self.period
        self.matcher_weight_dict = matcher_settings['weights_dict']
        self.matcher = TimeMatcher(**matcher_settings)

        set_crit_settings['box_coder'] = box_coder
        set_crit_settings['matcher'] = self.matcher
        self.set_crit = SetCriterion(**set_crit_settings)

        gt_processor_config = GT_PROCESSOR_CONFIG
        self.target_assigner = GroundTruthProcessor(gt_processor_cfg=gt_processor_settings)

        self.max_volumn_space = gt_processor_config['max_volumn_space']
        self.min_volumn_space = gt_processor_config['min_volumn_space']
        self.grid_size = gt_processor_config['grid_size']
        self.out_size_factor = out_size_factor
        self._generate_offset_grid()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        ret_dict = {}
        pred_centers = self.vote_head(x)
        pred_vote_cls = self.vote_cls_head(x)
        ret_dict['pred_centers'] = pred_centers
        ret_dict['pred_vote_cls'] = pred_vote_cls
        pos_embed = self.offset_grid.repeat(x.shape[0], 1, 1, 1)  # cart embed (1, 2, h, w) -> (B, 2, h, w)
        voted_embed = torch.cat([pred_centers, pred_vote_cls], dim=1)  # (B, 3, h, w)
        feat = self.layer(x, pos_embed, voted_embed)[0]

        pred_hm = self.cls_head(feat)
        pred_boxes = self.bbox_head(feat)

        ret_dict['hm'] = pred_hm
        ret_dict['reg'] = pred_boxes[:, :2, :, :]
        ret_dict['height'] = pred_boxes[:, 2:3, :, :]
        ret_dict['dim'] = pred_boxes[:, 3:6, :, :]
        ret_dict['rot'] = pred_boxes[:, 6:8, :, :]
        if self.iou_loss:
            ret_dict['iou'] = self.iou_head(feat)

        return {'det_preds': [ret_dict]}
    
    def _generate_offset_grid(self):
        x, y = self.grid_size[:2] // self.out_size_factor
        xmin, ymin, zmin = self.min_volumn_space
        xmax, ymax, zmax = self.max_volumn_space

        xoffset = (xmax - xmin) / x 
        yoffset = (ymax - ymin) / y

        yv, xv = torch.meshgrid([torch.arange(y), torch.arange(x)])
        yv = (yv.float() + 0.5) * yoffset + ymin
        xv = (xv.float() + 0.5) * xoffset + xmin

        cartx = xv * torch.cos(yv)
        carty = xv * torch.sin(yv)

        self.register_buffer('offset_grid', torch.stack([cartx, carty], dim=0)[None])
        self.register_buffer('xy_offset', torch.tensor([xoffset, yoffset]).view(1, 2, 1, 1))

    def get_proper_xy(self, pred_boxes):
        tmp_x, tmp_y, res = pred_boxes[:, 0, :, :], pred_boxes[:, 1, :, :], pred_boxes[:, 2:, :, :]
        tmp = torch.stack([tmp_x, tmp_y], dim=1)
        tmp = tmp + self.offset_grid
        return torch.cat([tmp, res], dim=1)
    
    def _sigmoid(self, x):
        y = torch.clamp(x.sigmoid_(), min=1e-4, max=1-1e-4)
        return y
    
    def loss(self, example, preds_dicts, **kwargs):
        rets = []

        target_boxes = example['global_box']
        target_boxes = target_boxes[..., [0, 1, 2, 3, 4, 5, -2, -1]]  # remove vel target
        task_gt_dicts = self.target_assigner.process(target_boxes)

        for task_id, preds_dict in enumerate(preds_dicts['det_preds']):
            if 'vel' in preds_dict:
                preds_dict['anno_box'] = torch.cat(
                    [preds_dict['reg'], preds_dict['height'], preds_dict['dim'], preds_dict['rot'], preds_dict['vel']],
                    dim=1)
            else:
                preds_dict['anno_box'] = torch.cat(
                    [preds_dict['reg'], preds_dict['height'], preds_dict['dim'], preds_dict['rot']], dim=1)

            pred_boxes = self.get_proper_xy(preds_dict['anno_box'])
            pred_centers = self.get_proper_xy(preds_dict['pred_centers'])

            bs, code, h, w = pred_boxes.size()
            pred_boxes = pred_boxes.permute(0, 2, 3, 1).view(bs, h * w, code)

            pred_logits = preds_dict['hm']
            _, cls, _, _ = pred_logits.size()
            pred_logits = pred_logits.permute(0, 2, 3, 1).view(bs, h * w, cls)

            pred_centers = pred_centers.permute(0, 2, 3, 1).view(bs, h * w, -1)
            pred_vote_cls = preds_dict['pred_vote_cls']
            pred_vote_cls = pred_vote_cls.permute(0, 2, 3, 1).view(bs, h * w, -1)

            task_pred_dicts = {
                'pred_logits': pred_logits,
                'pred_boxes': pred_boxes,
                'pred_centers': pred_centers,
                'pred_vote_cls': pred_vote_cls
            }

            if self.iou_loss:
                pred_iou = preds_dict['iou'].permute(0, 2, 3, 1).view(bs, h * w, -1)
                task_pred_dicts.update({'pred_ious': pred_iou})

            task_loss_dicts = self.set_crit(task_pred_dicts, task_gt_dicts[task_id])
            loss = task_loss_dicts['loss']

            ret = {}
            ret.update({'det_loss': loss, 'ce_loss': task_loss_dicts['loss_ce'].detach().cpu(),
                        'bbox_loss': task_loss_dicts['loss_bbox'].detach().cpu(),
                        'vote_reg_loss': task_loss_dicts['loss_vote'].detach().cpu(),
                        'vote_cls_loss': task_loss_dicts['loss_vote_cls'].detach().cpu()})

            if self.iou_loss:
                ret.update({'iou_loss': task_loss_dicts['loss_iou'].detach().cpu()})

            rets.append(ret)

        rets_merged = defaultdict(list)
        for ret in rets:
            for k, v in ret.items():
                rets_merged[k].append(v)
        return rets_merged

    @torch.no_grad()
    def predict(self, example, preds_dicts, test_cfg, **kwargs):
        rets = []
        metas = []

        double_flip = test_cfg.get('double_flip', False)

        post_center_range = test_cfg.post_center_limit_range
        if len(post_center_range) >0:
            post_center_range = torch.tensor(post_center_range, dtype=preds_dict['det_preds'][0]['hm'].dtype, device=preds_dict['det_preds'][0]['hm'].device)

        for task_id, preds_dict in enumerate(preds_dicts['det_preds']):
            for key,val in preds_dict.items():
                preds_dict[key] = val.permute(0, 2, 3, 1).contiguous()
            
            meta_list = example['metadata']
            metas.append(meta_list)

            batch_box_preds, batch_hm = self.decode(preds_dict, double_flip, test_cfg)

            prev_dets = kwargs.get('prev_dets', None)
            prev_det = None if prev_dets is None else prev_dets[task_id]
            sec_id = kwargs.get('sec_id', 0)

            rets.append(self.post_processing(batch_box_preds, batch_hm, test_cfg, post_center_range, prev_det, sec_id))

        if test_cfg.get('stateful_nms', False) or test_cfg.get('panoptic', False):
            return rets
        
        num_samples = len(rets[0])

        ret_list = []

        for i in range(num_samples):
            ret = {}
            for k in rets[0][i].keys():
                if k in ['box3d_lidar', 'scores']:
                    ret[k] = torch.cat([ret[i][k] for ret in rets])
                elif k in ['label_preds']:
                    flag = 0
                    for j, num_class in enumerate(self.num_classes):
                        rets[j][i][k] += flag
                        flag += num_class
                    ret[k] = torch.cat([ret[i][k] for ret in rets])
            
            ret['metadata'] = metas[0][i]
            ret_list.append(ret)
        
        return ret_list
    
    @torch.no_grad()
    def decode(self, preds_dict, double_flip, test_cfg):
        if not double_flip:
            preds_dict['hm'] = torch.sigmoid(preds_dict['hm'])
            preds_dict['dim'] = torch.exp(preds_dict['dim'])
        batch_rot = torch.atan2(preds_dict['rot'][..., 1:2], preds_dict['rot'][..., 0:1])
            
        batch, H, W, num_cls = preds_dict['hm'].size()

        batch_hei = preds_dict['height'].reshape(batch, H * W, 1)

        batch_rot = batch_rot.reshape(batch, H * W, 1)

        batch_dim = preds_dict['dim'].reshape(batch, H * W, 3)
        batch_hm = preds_dict['hm'].reshape(batch, H * W, num_cls)

        if self.iou_loss:
            batch_iou = preds_dict['iou'].reshape(batch, H * W, 1)
            batch_iou = (batch_iou + 1) * 0.5
            batch_iou = torch.clamp(batch_iou, min=0.0, max=1.0)
            iou_factor = torch.LongTensor([self.iou_factor]).to(batch_iou)
            batch_iou = torch.pow(batch_iou, iou_factor)
            batch_hm = batch_hm * batch_iou

        batch_reg = self.get_proper_xy(preds_dict['reg'].permute(0, 3, 1, 2).contiguous()).permute(0, 2, 3, 1).reshape(
            batch, H * W, 2)

        if self.voxel_shape == 'cuboid':
            raise NotImplementedError
        else:
            if test_cfg.get('rectify', False):
                assert self.box_coder.rectify
                xs, ys = batch_reg[..., 0], batch_reg[..., 1]
                azs = torch.atan2(ys, xs)
                azs = azs.reshape(batch, H * W, 1)
                batch_rot += azs
                batch_rot_ind = batch_rot.new_zeros(tuple(batch_rot.shape))
                batch_rot_ind[batch_rot > np.pi] = -1
                batch_rot_ind[batch_rot < - np.pi] = 1
                batch_rot_ind *= 2 * np.pi
                batch_rot += batch_rot_ind

        if 'vel' in preds_dict:
            raise NotImplementedError
        else:
            batch_box_preds = torch.cat([batch_reg, batch_hei, batch_dim, batch_rot], dim=2)
        return batch_box_preds, batch_hm
    
    @torch.no_grad()
    def post_processing(self, batch_box_preds, batch_hm, test_cfg, post_center_range, prev_det, sec_id):
        batch_size = len(batch_hm)

        stateful_nms = test_cfg.get('stateful_nms', False)
        panoptic = test_cfg.get('panoptic', False)
        prediction_dicts = []
        for i in range(batch_size):
            box_preds = batch_box_preds[i]
            hm_preds = batch_hm[i]
            
            scores, labels = torch.max(hm_preds, dim=-1)

            score_mask = scores > test_cfg.score_threshold
            distance_mask = (box_preds[..., :3] >= post_center_range[:3]).all(1) \
                & (box_preds[..., :3] <= post_center_range[3:]).all(1)

            mask = distance_mask & score_mask 

            box_preds = box_preds[mask]
            scores = scores[mask]
            labels = labels[mask]
            
            if stateful_nms:
                if self.voxel_shape == 'cuboid':
                    angle = 2 * np.pi / test_cfg.interval * sec_id
                else:
                    angle = test_cfg.interval * sec_id
                rot_sin = np.sin(-angle)
                rot_cos = np.cos(angle)
                rot_mat_T = torch.tensor([[rot_cos, -rot_sin], [rot_sin, rot_cos]], dtype=torch.float,
                                         device=box_preds.device)
                box_preds[:, :2] = box_preds[:, :2] @ rot_mat_T
                box_preds[:, -1] -= angle
                if box_preds.shape[1] > 7:
                    box_preds[:, 6:8] = box_preds[:, 6:8] @ rot_mat_T

                if prev_det is not None:
                    box_preds = torch.cat((prev_det[i]["box3d_lidar"], box_preds))
                    scores = torch.cat((prev_det[i]["scores"], scores))
                    labels = torch.cat((prev_det[i]["label_preds"], labels))
                if panoptic:
                    sec_ids = labels.new_zeros((len(labels, )), dtype=int)
                    sec_ids[:len(prev_det[i]["box3d_lidar"])] = sec_id - 1
                    sec_ids[len(prev_det[i]["box3d_lidar"]):] = sec_id
                    instances = labels.new_zeros((len(labels, )), dtype=int)
                    instances[:len(prev_det[i]["box3d_lidar"])] = prev_det[i]["instances"]
                    if len(instances):
                        offset = len(prev_det[i]["box3d_lidar"]) + 1
                
            boxes_for_nms = box_preds[:, [0, 1, 2, 3, 4, 5, -1]]

            nms_post_max_size = test_cfg.nms.nms_post_max_size

            if test_cfg.get('per_class_nms', False):
                boxes_for_nms = boxes_for_nms[:, [0, 1, 3, 4, -1]]
                boxes_for_nms[:, -1] = boxes_for_nms[:, -1] / np.pi * 180
                selected = layers.batched_nms_rotated(boxes_for_nms, scores, labels, test_cfg.nms.nms_iou_threshold)
                selected = selected[:nms_post_max_size]
            else:
                selected = box_torch_ops.rotate_nms_pcdet(boxes_for_nms, scores, 
                                    thresh=test_cfg.nms.nms_iou_threshold,
                                    pre_maxsize=test_cfg.nms.nms_pre_max_size,
                                    post_max_size=nms_post_max_size)
                # boxes_for_nms = boxes_for_nms[:, [0, 1, 3, 4, -1]]
                # boxes_for_nms[:, -1] = boxes_for_nms[:, -1] / np.pi * 180
                # selected = layers.nms_rotated(boxes_for_nms, scores, test_cfg.nms.nms_iou_threshold)
                # selected = selected[:nms_post_max_size]
            selected_boxes = box_preds[selected]
            selected_scores = scores[selected]
            selected_labels = labels[selected]               

            if (not stateful_nms) and (sec_id > 0):
                if self.voxel_shape == 'cuboid':
                    angle = 2 * np.pi / test_cfg.interval * sec_id
                else:
                    angle = test_cfg.interval * sec_id
                rot_sin = np.sin(-angle)
                rot_cos = np.cos(angle)
                rot_mat_T = torch.tensor([[rot_cos, -rot_sin],[rot_sin, rot_cos]], dtype=torch.float, device=selected_boxes.device)
                selected_boxes[:, :2] = selected_boxes[:, :2] @ rot_mat_T
                selected_boxes[:, -1] -= angle
                if selected_boxes.shape[1] > 7:
                    selected_boxes[:, 6:8] = selected_boxes[:, 6:8] @ rot_mat_T

            prediction_dict = {
                'box3d_lidar': selected_boxes,
                'scores': selected_scores,
                'label_preds': selected_labels
            }
            if panoptic:
                if sec_id == 0:
                    instances = torch.arange(len(selected_boxes), device=selected_boxes.device)
                elif stateful_nms:
                    instances = instances[selected]
                    if len(instances):
                        sec_ids = sec_ids[selected]
                        mask = sec_ids == sec_id
                        tmp_id = torch.arange(mask.sum(), device=selected_boxes.device)
                        instances[mask] = tmp_id + offset
                else:
                    if len(prev_det[i]["box3d_lidar"]):
                        prediction_dict['box3d_lidar'] = torch.cat((prev_det[i]["box3d_lidar"], selected_boxes))
                        prediction_dict['scores'] = torch.cat((prev_det[i]['scores'], selected_scores))
                        prediction_dict['label_preds'] = torch.cat((prev_det[i]['label_preds'], selected_labels))
                        offset = len(prev_det[i]['instances'])
                        instances = torch.arange(len(selected_boxes), device=selected_boxes.device) + offset
                        instances = torch.cat((prev_det[i]['instances'], instances))
                    else:
                        instances = torch.arange(len(selected_boxes), device=selected_boxes.device)
                
                prediction_dict.update({'instances': instances})

            prediction_dicts.append(prediction_dict)
        return prediction_dicts
