from collections import defaultdict
import torch
from torch import nn
import torch.distributed as dist
from .loss_utils import E2ESigmoidFocalClassificationLoss, SmoothL1Loss, IOULoss, IouRegLoss


def is_distributed():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_distributed():
        return 1
    return dist.get_world_size()


def label_to_one_hot(label, pred):
    label = label.unsqueeze(-1)
    bs, querys, cls = pred.size()
    one_hot = torch.full((bs, querys, cls + 1), 0, dtype=torch.float32, device=pred.device)
    one_hot.scatter_(dim=-1, index=label, value=1.0)
    return one_hot[..., 1:]


class SetCriterion(nn.Module):
    def __init__(self, matcher, weight_dict, losses, sigma, box_coder, code_weights,
                 gamma=2.0, alpha=0.25, iou_reg_type='DIoU', **kwargs):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.register_buffer('code_weights', torch.Tensor(code_weights))

        self.loss_map = {
            'loss_ce': self.loss_labels,
            'loss_bbox': self.loss_boxes,
            'loss_vote': self.loss_vote,
            'loss_vote_cls': self.loss_vote_cls,
            'loss_iou': self.loss_iou,
            'loss_iou_reg': self.loss_iou_reg
        }

        self.box_coder = box_coder
        self.focal_loss = E2ESigmoidFocalClassificationLoss(gamma=gamma, alpha=alpha, reduction='sum')
        self.reg_loss = SmoothL1Loss(sigma=sigma)
        self.iou_loss = IOULoss()
        self.iou_reg_loss = IouRegLoss(iou_reg_type)

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _preprocess(self, pred_dicts, gt_dicts, **kwargs):
        conds = defaultdict(lambda: None)
        # conds['cur_epoch'] = cur_epoch

        matcher_dict = self.matcher(pred_dicts, gt_dicts)
        indices = matcher_dict['inds']
        idx = self._get_src_permutation_idx(indices)

        pred_boxes = pred_dicts['pred_boxes'][idx]

        if 'pred_ious' in pred_dicts.keys():
            pred_iou = pred_dicts['pred_ious'][idx]
        if hasattr(self.box_coder, 'grid_offset'):
            xy_grid = self.box_coder.grid_offset.flatten(2).permute(0, 2, 1).repeat(2, 1, 1)
            xy_grid = xy_grid[idx]
            x_grid, y_grid = xy_grid[..., 0], xy_grid[..., 1]
        else:
            x_grid = None
            y_grid = None
        gt_boxes = torch.cat([iter[i] for iter, (_, i) in zip(gt_dicts['gt_boxes'], indices)], dim=0)

        if self.box_coder is not None:
            tmp_delta = self.box_coder.get_delta(gt_boxes=gt_boxes, preds=pred_boxes, x_grid=x_grid, y_grid=y_grid)
        else:
            tmp_delta = gt_boxes - pred_boxes

        tmp_delta = torch.einsum('bc,c->bc', tmp_delta, self.code_weights)
        conds['delta'] = tmp_delta
        conds['gt_boxes'] = gt_boxes[:, :7]
        conds['pred_boxes'] = self.box_coder.decode_torch(preds=pred_boxes)[:, :7]

        if 'pred_ious' in pred_dicts.keys():
            conds['pred_ious'] = pred_iou

        # preprocess for cls loss
        pred_logits = pred_dicts['pred_logits']
        gt_classes_pos = torch.cat([t[j] for t, (_, j) in zip(gt_dicts['gt_classes'], indices)]) + 1
        gt_classes = torch.full(pred_logits.shape[:2], 0, dtype=torch.int64, device=pred_logits.device)
        gt_classes[idx] = gt_classes_pos

        conds['pred_logits'] = pred_logits
        conds['gt_classes'] = gt_classes

        # preprocess for cart loss
        target_len = torch.as_tensor([len(iter) for iter in gt_dicts['gt_classes']], device=pred_logits.device)
        conds['target_len'] = target_len

        num_boxes = sum([len(iter) for iter in gt_dicts['gt_classes']])
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=pred_logits.device)

        if is_distributed():
            torch.distributed.all_reduce(num_boxes)

        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()
        conds['num_boxes'] = num_boxes

        # preprocess for vote loss
        if 'pred_centers' in pred_dicts.keys():
            b, l, c = pred_dicts['pred_centers'].shape
            assert c in [2, 4]
            votemap = gt_dicts['votemap'].reshape(b, l, -1)
            votemask = votemap[:, :, 0] != 0

            vote_num = torch.sum(votemask)
            # vote_num = torch.clamp(vote_num / get_world_size(), min=1).item()
            vote_num = torch.clamp(vote_num, min=1).item()
            conds['vote_num'] = vote_num
            conds['pred_centers'] = pred_dicts['pred_centers'][votemask]
            conds['gt_centers'] = votemap[:, :, :c][votemask]

            conds['pred_vote_logits'] = pred_dicts['pred_vote_cls']
            vote_cls = votemap[:, :, 4:]
            conds['gt_logits'] = vote_cls

        return conds

    def loss_boxes(self, delta, num_boxes, **kwargs):
        loss_bbox_loc = self.reg_loss(delta)
        loss_bbox = loss_bbox_loc.sum() / num_boxes
        loss_bbox_loc = loss_bbox_loc.detach().clone() / num_boxes

        losses = {
            'loss_bbox': loss_bbox,
            'loc_loss_elem': loss_bbox_loc
        }
        return losses

    def loss_vote(self, pred_centers, gt_centers, vote_num, **kwargs):
        delta = pred_centers - gt_centers
        loss_vote = self.reg_loss(delta)
        loss_vote = loss_vote.sum() / vote_num

        losses = {
            'loss_vote': loss_vote
        }
        return losses

    def loss_vote_cls(self, pred_vote_logits, gt_logits, vote_num, **kwargs):
        loss_vote_cls = self.focal_loss(pred_vote_logits, gt_logits) / vote_num
        losses = {'loss_vote_cls': loss_vote_cls}
        return losses

    def loss_labels(self, pred_logits, gt_classes, num_boxes, **kwargs):
        """
        Classification loss (NLL)
        """
        target_one_hot = label_to_one_hot(gt_classes, pred_logits)
        loss_ce = self.focal_loss(pred_logits, target_one_hot) / num_boxes

        losses = {'loss_ce': loss_ce}
        return losses

    def loss_iou(self, pred_boxes, pred_ious, gt_boxes, num_boxes, **kwargs):
        loss_iou = self.iou_loss(pred_boxes, pred_ious, gt_boxes) / num_boxes
        losses = {
            'loss_iou': loss_iou
        }
        return losses

    def loss_iou_reg(self, pred_boxes, gt_boxes, num_boxes, **kwargs):
        loss_iou_reg = self.iou_reg_loss(pred_boxes, gt_boxes) / num_boxes
        losses = {
            'loss_iou_reg': loss_iou_reg
        }
        return losses

    def get_loss(self, loss, conds):
        assert loss in self.loss_map, f'do you really want to compute {loss} loss?'
        return self.loss_map[loss](**conds)

    def forward(self, pred_dicts, gt_dicts):
        conds = self._preprocess(pred_dicts, gt_dicts)
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, conds))

        total = sum([losses[k] * self.weight_dict[k] for k in losses.keys() if k in self.weight_dict])
        losses['loss'] = total
        return losses
