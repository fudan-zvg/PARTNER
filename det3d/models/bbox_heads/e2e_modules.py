import torch
import math
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import det3d.models.e2e_utils.centernet_utils as centernet_utils
from det3d.core.bbox import box_torch_ops
from det3d.models.e2e_utils.common_utils import limit_period_torch, mesh_formulation_torch


class GroundTruthProcessor(object):
    def __init__(self, gt_processor_cfg):
        self.tasks = gt_processor_cfg.tasks
        self.class_to_idx = gt_processor_cfg.mapping
        self.period = 2 * np.pi

        self.generate_votemap = gt_processor_cfg.get('generate_votemap', False)

        self.max_volumn_space = gt_processor_cfg.get('max_volumn_space', [100, 3.1415926, 4])
        self.min_volumn_space = gt_processor_cfg.get('min_volumn_space', [ 0, -3.1415926, -2])
        self.grid_size = np.array(gt_processor_cfg.get('grid_size', [1440, 2520, 40]))
        self.feature_map_stride = gt_processor_cfg.get('feature_map_stride', 8)
        self.discret_mode = gt_processor_cfg.get('discret_mode', 'UD')
        self.gaussian_overlap = gt_processor_cfg.get('gaussian_overlap', 0.1)
        self.min_radius = gt_processor_cfg.get('min_radius', 4)
        self.num_max_objs = gt_processor_cfg.get('num_max_objs', 500)
        self.scale_factor = gt_processor_cfg.get('scale_factor', 1)

        self.feature_map_size = self.grid_size[::-1] / self.feature_map_stride

    def process(self, gt_boxes):
        """
        Args:
            gt_boxes: (B, M, C + cls)

        Returns:
            gt_dicts: a dict key is task id
            each item is a dict {
                gt_class: list(Tensor), len = batchsize, Tensor with size (box_num, 10)
                gt_boxes: list(Tensor), len = batchsize, Tensor with size (box_num, 10)
            }
        """

        batch_size = gt_boxes.shape[0]
        gt_classes = gt_boxes[:, :, -1]  # begin from 1
        gt_boxes = gt_boxes[:, :, :-1]

        gt_dicts = {}

        for task_id, task in enumerate(self.tasks):
            gt_dicts[task_id] = {}
            gt_dicts[task_id]['gt_classes'] = []
            gt_dicts[task_id]['gt_boxes'] = []
            gt_dicts[task_id]['votemap'] = []

        for k in range(batch_size):
            # remove padding
            iter_box = gt_boxes[k]
            count = len(iter_box) - 1
            while count > 0 and iter_box[count].sum() == 0:
                count -= 1

            iter_box = iter_box[:count + 1]
            iter_gt_classes = gt_classes[k][:count + 1].int()

            for task_id, task in enumerate(self.tasks):
                boxes_of_tasks = []
                classes_of_tasks = []
                class_offset = 0

                for class_name in task.class_names:
                    class_idx = self.class_to_idx[class_name]
                    class_mask = (iter_gt_classes == class_idx)
                    _boxes = iter_box[class_mask]
                    _class = _boxes.new_full((_boxes.shape[0],), class_offset).long()
                    boxes_of_tasks.append(_boxes)
                    classes_of_tasks.append(_class)
                    class_offset += 1

                task_boxes = torch.cat(boxes_of_tasks, dim=0)
                task_classes = torch.cat(classes_of_tasks, dim=0)
                gt_dicts[task_id]['gt_boxes'].append(task_boxes)
                gt_dicts[task_id]['gt_classes'].append(task_classes)
                gt_dicts[task_id]['gt_cls_num'] = len(task.class_names)

                if self.generate_votemap:
                    votemap = self.draw_votemap(task_boxes, task_classes, len(task.class_names))  # vote center both cart and polar
                    gt_dicts[task_id]['votemap'].append(votemap)

        if self.generate_votemap:
            for task_id, task in enumerate(self.tasks):
                gt_dicts[task_id]['votemap'] = torch.stack(gt_dicts[task_id]['votemap'], dim=0)

        return gt_dicts

    def draw_votemap(self, task_boxes, task_classes, num_class):
        votemap = task_boxes.new_zeros(int(self.feature_map_size[1]), int(self.feature_map_size[2]), 4+num_class)

        if task_boxes.shape[0] == 0:
            return votemap

        task_corners = box_torch_ops.center_to_corner_box2d(task_boxes[:, :2], task_boxes[:, 3:5], angles=task_boxes[:, 6])
        corners_rhos, corners_phis = torch.norm(task_corners, 2, 2), torch.atan2(task_corners[:, :, 1], task_corners[:, :, 0])

        max_rho, _ = torch.max(corners_rhos, dim=1)
        min_rho, _ = torch.min(corners_rhos, dim=1)
        max_phi, _ = torch.max(corners_phis, dim=1)
        min_phi, _ = torch.min(corners_phis, dim=1)

        max_bound = torch.from_numpy(np.asarray(self.max_volumn_space)).to(task_boxes.device)
        min_bound = torch.from_numpy(np.asarray(self.min_volumn_space)).to(task_boxes.device)
        grid_size = torch.from_numpy(np.asarray(self.grid_size)).to(task_boxes.device)
        voxel_size = [(max_bound[i] - min_bound[i]) / grid_size[i] for i in range(len(max_bound))]

        drho = (max_rho - min_rho) / voxel_size[0] / self.feature_map_stride
        dphi = (max_phi - min_phi) / voxel_size[1] / self.feature_map_stride

        center_xs = task_boxes[:, 0]
        center_ys = task_boxes[:, 1]
        center_rhos = torch.norm(task_boxes[:, :2], 2, 1)
        center_phis = torch.atan2(task_boxes[:, 1], task_boxes[:, 0])
        centers = torch.stack([center_xs, center_ys, center_rhos, center_phis], dim=-1)

        center_rhos_ind = (center_rhos - min_bound[0]) / voxel_size[0] / self.feature_map_stride
        center_phis_ind = (center_phis - min_bound[1]) / voxel_size[1] / self.feature_map_stride
        centers_ind = torch.stack([center_rhos_ind, center_phis_ind], dim=-1)
        centers_ind_int = centers_ind.int()

        for k in range(min(self.num_max_objs, task_boxes.shape[0])):
            if drho[k] <= 0 or dphi[k] <= 0:
                continue

            if not (0 <= centers_ind_int[k][0] < self.feature_map_size[2] and 0 <= centers_ind_int[k][1] <
                    self.feature_map_size[
                        1]):
                continue

            # trunc
            if dphi[k] > (self.feature_map_size[1] / 4):
                corners_phis_spec = corners_phis[k]
                if torch.atan2(task_boxes[k, 1], task_boxes[k, 0]) > 0:
                    trunc_phi = math.pi - torch.min(corners_phis_spec[corners_phis_spec > 0])
                else:
                    trunc_phi = torch.max(corners_phis_spec[corners_phis_spec <= 0]) + math.pi
                dphi[k] = trunc_phi / voxel_size[1] / self.feature_map_stride

            centernet_utils.draw_center_to_votemap(votemap, centers_ind_int[k], centers[k], drho[k], dphi[k], task_classes[k])
        return votemap
