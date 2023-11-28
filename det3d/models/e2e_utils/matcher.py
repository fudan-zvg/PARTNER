import torch
import numpy as np
import torch.nn as nn
from scipy.optimize import linear_sum_assignment
from collections import defaultdict


class TimeMatcher(nn.Module):
    def __init__(
            self,
            box_coder,
            losses,
            weight_dict,
            use_focal_loss,
            code_weights,
            period=None,
            iou_th=-1,
            **kwargs
    ):
        super().__init__()
        self.losses = losses
        self.box_coder = box_coder  # notice here we also need encode ground truth boxes
        self.weight_dict = weight_dict
        self.period = period
        self.iou_th = iou_th
        self.register_buffer('code_weights', torch.Tensor(code_weights))
        self.use_focal_loss = use_focal_loss
        self.loss_dict = {
            'loss_ce': self.loss_ce,
            'loss_bbox': self.loss_bbox,
        }

    @staticmethod
    def _prep(input):
        slices = [input[..., :3], input[..., 3:6], input[..., 6:]]
        slices[1] = torch.clamp_min(slices[1], min=1e-5)
        return torch.cat(slices, dim=-1)

    def _preprocess(self, pred_dicts, gt_dicts):
        '''
        Args:
            pred_dicts: {
                pred_logits: list(Tensor), Tensor with size (box_num, cls_num)
                pred_boxes: list(Tensor), Tensor with size (box_num, 10)
            }
            gt_dicts: {
                gt_class: list(Tensor), Tensor with size (box_num)
                gt_boxes: list(Tensor), Tensor with size (box_num, 10)
            }

        Returns:

        '''

        examples = defaultdict(lambda: None)

        examples['pred_boxes'] = pred_dicts['pred_boxes']
        examples['gt_boxes'] = self.box_coder.encode(gt_dicts['gt_boxes'])

        if self.iou_th > 0.0:
            examples['pred_boxes_iou'] = self.box_coder.decode_torch(pred_dicts['pred_boxes'])
            examples['gt_boxes_iou'] = gt_dicts['gt_boxes']

        pred_logits = pred_dicts['pred_logits']
        if self.use_focal_loss:
            pred_logits = pred_logits.sigmoid()
        else:
            pred_logits = pred_logits.softmax(dim=-1)

        examples['pred_logits'] = pred_logits
        examples['gt_classes'] = gt_dicts['gt_classes']
        examples['batchsize'] = pred_dicts['pred_boxes'].size(0)
        examples['code_weights'] = pred_dicts.get('code_weights', None)

        return examples

    def loss_ce(self, pred_logits, gt_classes, **kwargs):
        loss = pred_logits[:, gt_classes]
        loss[loss == float("Inf")] = 0
        loss[loss != loss] = 0
        return loss

    def loss_bbox(self, pred_boxes, gt_boxes, code_weights=None, **kwargs):
        if code_weights is None:
            code_weights = self.code_weights

        weighted_preds = torch.einsum('bc,c->bc', pred_boxes, code_weights)
        weighted_gts = torch.einsum('bc,c->bc', gt_boxes, code_weights)
        loss = torch.exp(-torch.cdist(weighted_preds, weighted_gts, p=1))
        loss[loss == float("Inf")] = 0
        loss[loss != loss] = 0

        return loss

    def get_loss(self, example):
        rlt = {}

        if self.iou_th > 0.0:
            rlt['iou_thresh_mask'] = self.iou_thresh(**example)

        for k in self.losses:
            if k not in self.weight_dict:
                continue
            rlt[k] = self.loss_dict[k](**example)
        return rlt

    def _get_per_scene_example(self, examples, idx):
        rlt = {}

        rlt['pred_logits'] = examples['pred_logits'][idx]
        rlt['pred_boxes'] = examples['pred_boxes'][idx]

        rlt['gt_classes'] = examples['gt_classes'][idx]
        rlt['gt_boxes'] = examples['gt_boxes'][idx]
        rlt['code_weights'] = examples.get('code_weights', None)

        if self.iou_th > 0.0:
            rlt['pred_boxes_iou'] = examples['pred_boxes_iou'][idx]
            rlt['gt_boxes_iou'] = examples['gt_boxes_iou'][idx]

        return rlt

    @torch.no_grad()
    def forward(self, pred_dicts, gt_dicts):
        rlt = {}
        examples = self._preprocess(pred_dicts, gt_dicts)
        indices = []

        for i in range(examples['batchsize']):

            if examples["gt_classes"][i].size(0) == 0:
                indices.append(([], []))
                continue

            example = self._get_per_scene_example(examples, i)
            loss_val_dict = self.get_loss(example)

            loss = -1.0
            if self.iou_th > 0.0:
                loss = loss * loss_val_dict['iou_thresh_mask']
            for k in self.losses:
                if k not in self.weight_dict:
                    continue
                tmp = loss_val_dict[k] ** self.weight_dict[k]
                loss = loss * tmp

            ind = linear_sum_assignment(loss.cpu())

            indices.append(ind)

        rlt['inds'] = [
            (torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices
        ]
        return rlt
