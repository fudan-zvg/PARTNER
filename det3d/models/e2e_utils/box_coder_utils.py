import numpy as np
import torch


def check_numpy_to_torch(x):
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).float(), True
    return x, False

def rotate_points_along_z(points, angle):
    """
    Args:
        points: (B, N, 3 + C)
        angle: (B), angle along z-axis, angle increases x ==> y
    Returns:

    """
    points, is_numpy = check_numpy_to_torch(points)
    angle, _ = check_numpy_to_torch(angle)

    cosa = torch.cos(angle)
    sina = torch.sin(angle)
    zeros = angle.new_zeros(points.shape[0])
    ones = angle.new_ones(points.shape[0])
    rot_matrix = torch.stack((
        cosa, sina, zeros,
        -sina, cosa, zeros,
        zeros, zeros, ones
    ), dim=1).view(-1, 3, 3).float()
    points_rot = torch.matmul(points[:, :, 0:3], rot_matrix)
    points_rot = torch.cat((points_rot, points[:, :, 3:]), dim=-1)
    return points_rot.numpy() if is_numpy else points_rot


def boxes_to_corners_3d(boxes3d):
    """
        7 -------- 4
       /|         /|
      6 -------- 5 .
      | |        | |
      . 3 -------- 0
      |/         |/
      2 -------- 1
    Args:
        boxes3d:  (N, 7) [x, y, z, dx, dy, dz, heading], (x, y, z) is the box center

    Returns:
    """
    boxes3d, is_numpy = check_numpy_to_torch(boxes3d)

    template = boxes3d.new_tensor((
        [1, 1, -1], [1, -1, -1], [-1, -1, -1], [-1, 1, -1],
        [1, 1, 1], [1, -1, 1], [-1, -1, 1], [-1, 1, 1],
    )) / 2

    corners3d = boxes3d[:, None, 3:6].repeat(1, 8, 1) * template[None, :, :]
    corners3d = rotate_points_along_z(corners3d.view(-1, 8, 3), boxes3d[:, 6]).view(-1, 8, 3)
    corners3d += boxes3d[:, None, 0:3]

    return corners3d.numpy() if is_numpy else corners3d


# this coder is only for e2e cases which has quite different logic
class CenterCoder(object):
    def __init__(self, code_size=7, encode_angle_by_sincos=False, period=2 * np.pi, rectify=False, **kwargs):
        super().__init__()
        self.code_size = code_size
        self.encode_angle_by_sincos = encode_angle_by_sincos
        self.period = period
        self.rectify = rectify
        if self.encode_angle_by_sincos:
            self.code_size += 1

    @staticmethod
    def _rotate_points_along_z(points, angle):
        cosa = torch.cos(angle)
        sina = torch.sin(angle)
        zeros = angle.new_zeros(points.shape[0])
        ones = angle.new_ones(points.shape[0])
        rot_matrix = torch.stack((
            cosa, sina, zeros,
            -sina, cosa, zeros,
            zeros, zeros, ones
        ), dim=1).view(-1, 3, 3).float()
        points_rot = torch.matmul(points[:, :, 0:3], rot_matrix)
        points_rot = torch.cat((points_rot, points[:, :, 3:]), dim=-1)
        return points_rot

    @staticmethod
    def boxes_to_corners_3d(boxes3d):
        template = boxes3d.new_tensor((
            [1, 1, -1], [1, -1, -1], [-1, -1, -1], [-1, 1, -1],
            [1, 1, 1], [1, -1, 1], [-1, -1, 1], [-1, 1, 1],
        )) / 2

        corners3d = boxes3d[:, None, 3:6].repeat(1, 8, 1) * template[None, :, :]
        corners3d = CenterCoder._rotate_points_along_z(corners3d.view(-1, 8, 3), boxes3d[:, 6]).view(-1, 8, 3)
        corners3d += boxes3d[:, None, 0:3]
        return corners3d

    @staticmethod
    def _prep(input):
        slices = [input[..., :3], input[..., 3:6], input[..., 6:]]
        slices[1] = torch.clamp_min(slices[1], min=1e-5)
        return torch.cat(slices, dim=-1)

    def encode(self, gt_boxes):
        rlt = []
        for i in range(len(gt_boxes)):
            tmp = CenterCoder._prep(gt_boxes[i])

            xg, yg, zg, dxg, dyg, dzg, rg, *cgs = torch.split(tmp, 1, dim=-1)

            dxn = torch.log(dxg)
            dyn = torch.log(dyg)
            dzn = torch.log(dzg)

            # rectify: calculate gt center angle
            if self.rectify:
                phig = torch.atan2(yg, xg)
                rel_rg = rg - phig
                rel_rg_ind = rel_rg.new_zeros(tuple(rel_rg.shape))
                rel_rg_ind[rel_rg > np.pi] = -1
                rel_rg_ind[rel_rg < -np.pi] = 1
                rel_rg_ind *= 2 * np.pi
                rel_rg += rel_rg_ind
                rg = rel_rg

            if self.encode_angle_by_sincos:
                cosg = torch.cos(rg)
                sing = torch.sin(rg)
                rgs = [cosg, sing]
            else:
                rgs = [rg, ]

            rlt.append(torch.cat([xg, yg, zg, dxn, dyn, dzn, *rgs, *cgs], dim=-1))

        return rlt

    def gt_to_corner(self, gt_boxes):
        rlt = []
        for i in range(len(gt_boxes)):
            boxes3d = CenterCoder._prep(gt_boxes[i])
            corners3d = CenterCoder.boxes_to_corners_3d(boxes3d)
            rlt.append(corners3d)
        return rlt

    def pred_to_corner(self, preds):

        if not self.encode_angle_by_sincos:
            xt, yt, zt, dxt, dyt, dzt, rt, *cts = torch.split(preds, 1, dim=-1)
        else:
            xt, yt, zt, dxt, dyt, dzt, cost, sint, *cts = torch.split(preds, 1, dim=-1)
            rt = torch.atan2(sint, cost)

        dxg = torch.exp(dxt)
        dyg = torch.exp(dyt)
        dzg = torch.exp(dzt)

        all_boxes = torch.cat([xt, yt, zt, dxg, dyg, dzg], dim=-1)


        rlt = []
        for i in range(len(all_boxes)):
            boxes3d = all_boxes[i]
            corners3d = CenterCoder.boxes_to_corners_3d(boxes3d)
            rlt.append(corners3d)
            rlt.append(corners3d)

        return rlt

    def get_delta(self, gt_boxes, preds, **kwargs):
        """
        Args:
            gt_boxes: (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]
            gts: (N, 7 + C) [x, y, z, dx, dy, dz, heading or *[cos, sin], ...]

        Returns:

        """
        gt_boxes = CenterCoder._prep(gt_boxes)

        xg, yg, zg, dxg, dyg, dzg, rg, *cgs = torch.split(gt_boxes, 1, dim=-1)  # has been log
        if not self.encode_angle_by_sincos:
            xp, yp, zp, dxp, dyp, dzp, rp, *cps = torch.split(preds, 1, dim=-1)
        else:
            xp, yp, zp, dxp, dyp, dzp, cosp, sinp, *cps = torch.split(preds, 1, dim=-1)

        xt = xg - xp
        yt = yg - yp
        zt = zg - zp

        dxt = torch.log(dxg) - dxp
        dyt = torch.log(dyg) - dyp
        dzt = torch.log(dzg) - dzp

        # rectify: calculate gt center angle
        if self.rectify:
            phig = torch.atan2(yg, xg)
            rel_rg = rg - phig
            rel_rg_ind = rel_rg.new_zeros(tuple(rel_rg.shape))
            rel_rg_ind[rel_rg > np.pi] = -1
            rel_rg_ind[rel_rg < -np.pi] = 1
            rel_rg_ind *= 2 * np.pi
            rel_rg += rel_rg_ind
            rg = rel_rg

        if self.encode_angle_by_sincos:
            rt_cos = torch.cos(rg) - cosp
            rt_sin = torch.sin(rg) - sinp
            rts = [rt_cos, rt_sin]
        else:
            rts = [(rg / self.period) - rp]

        cts = [g - a for g, a in zip(cgs, cps)]
        return torch.cat([xt, yt, zt, dxt, dyt, dzt, *rts, *cts], dim=-1)

    def decode_torch(self, preds):
        """
        Args:
            box_encodings: (B, N, 7 + C) or (N, 7 + C) [x, y, z, dx, dy, dz, heading or *[cos, sin], ...]
        Returns:

        """
        if not self.encode_angle_by_sincos:
            xt, yt, zt, dxt, dyt, dzt, rt, *cts = torch.split(preds, 1, dim=-1)
        else:
            xt, yt, zt, dxt, dyt, dzt, cost, sint, *cts = torch.split(preds, 1, dim=-1)

        dxg = torch.exp(dxt)
        dyg = torch.exp(dyt)
        dzg = torch.exp(dzt)

        if self.rectify:
            raise NotImplementedError

        if self.encode_angle_by_sincos:
            rg = torch.atan2(sint, cost)
        else:
            rg = rt * self.period

        cgs = [t for t in cts]
        return torch.cat([xt, yt, zt, dxg, dyg, dzg, rg, *cgs], dim=-1)


    def encode_with_rois(self, boxes, rois):
        """
        Args:
            boxes: (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]
            anchors: (N, 7 + C) [x, y, z, dx, dy, dz, heading or *[cos, sin], ...]

        Returns:

        """
        rois[:, 3:6] = torch.clamp_min(rois[:, 3:6], min=1e-5)
        boxes[:, 3:6] = torch.clamp_min(boxes[:, 3:6], min=1e-5)

        xa, ya, za, dxa, dya, dza, ra, *cas = torch.split(rois, 1, dim=-1)
        xg, yg, zg, dxg, dyg, dzg, rg, *cgs = torch.split(boxes, 1, dim=-1)

        diagonal = torch.sqrt(dxa ** 2 + dya ** 2)
        xt = (xg - xa) / diagonal
        yt = (yg - ya) / diagonal
        zt = (zg - za) / dza
        dxt = torch.log(dxg / dxa)
        dyt = torch.log(dyg / dya)
        dzt = torch.log(dzg / dza)
        if self.encode_angle_by_sincos:
            rt_cos = torch.cos(rg) - torch.cos(ra)
            rt_sin = torch.sin(rg) - torch.sin(ra)
            rts = [rt_cos, rt_sin]
        else:
            rts = [rg - ra]

        cts = [g - a for g, a in zip(cgs, cas)]
        return torch.cat([xt, yt, zt, dxt, dyt, dzt, *rts, *cts], dim=-1)


# rectify, use init encode logic but different in cal delta, however, decode func is not used, instead, change matcher code weights in config
class CenterCoderV2(object):
    def __init__(self, code_size=7, encode_angle_by_sincos=False, period=2 * np.pi, rectify=False, **kwargs):
        super().__init__()
        self.code_size = code_size
        self.encode_angle_by_sincos = encode_angle_by_sincos
        self.period = period
        self.rectify = rectify
        self.grid_offset = None
        if self.encode_angle_by_sincos:
            self.code_size += 1

    @staticmethod
    def _rotate_points_along_z(points, angle):
        cosa = torch.cos(angle)
        sina = torch.sin(angle)
        zeros = angle.new_zeros(points.shape[0])
        ones = angle.new_ones(points.shape[0])
        rot_matrix = torch.stack((
            cosa, sina, zeros,
            -sina, cosa, zeros,
            zeros, zeros, ones
        ), dim=1).view(-1, 3, 3).float()
        points_rot = torch.matmul(points[:, :, 0:3], rot_matrix)
        points_rot = torch.cat((points_rot, points[:, :, 3:]), dim=-1)
        return points_rot

    @staticmethod
    def boxes_to_corners_3d(boxes3d):
        template = boxes3d.new_tensor((
            [1, 1, -1], [1, -1, -1], [-1, -1, -1], [-1, 1, -1],
            [1, 1, 1], [1, -1, 1], [-1, -1, 1], [-1, 1, 1],
        )) / 2

        corners3d = boxes3d[:, None, 3:6].repeat(1, 8, 1) * template[None, :, :]
        corners3d = CenterCoder._rotate_points_along_z(corners3d.view(-1, 8, 3), boxes3d[:, 6]).view(-1, 8, 3)
        corners3d += boxes3d[:, None, 0:3]
        return corners3d

    @staticmethod
    def _prep(input):
        slices = [input[..., :3], input[..., 3:6], input[..., 6:]]
        slices[1] = torch.clamp_min(slices[1], min=1e-5)
        return torch.cat(slices, dim=-1)

    def encode(self, gt_boxes):
        rlt = []
        for i in range(len(gt_boxes)):
            tmp = CenterCoder._prep(gt_boxes[i])

            xg, yg, zg, dxg, dyg, dzg, rg, *cgs = torch.split(tmp, 1, dim=-1)

            dxn = torch.log(dxg)
            dyn = torch.log(dyg)
            dzn = torch.log(dzg)

            # rectify: calculate gt center angle
            # if self.rectify:
            #     phig = torch.atan2(yg, xg)
            #     rel_rg = rg - phig
            #     rel_rg_ind = rel_rg.new_zeros(tuple(rel_rg.shape))
            #     rel_rg_ind[rel_rg > np.pi] = -1
            #     rel_rg_ind[rel_rg < -np.pi] = 1
            #     rel_rg_ind *= 2 * np.pi
            #     rel_rg += rel_rg_ind
            #     rg = rel_rg

            if self.encode_angle_by_sincos:
                cosg = torch.cos(rg)
                sing = torch.sin(rg)
                rgs = [cosg, sing]
            else:
                rgs = [rg, ]

            rlt.append(torch.cat([xg, yg, zg, dxn, dyn, dzn, *rgs, *cgs], dim=-1))

        return rlt

    def gt_to_corner(self, gt_boxes):
        rlt = []
        for i in range(len(gt_boxes)):
            boxes3d = CenterCoder._prep(gt_boxes[i])
            corners3d = CenterCoder.boxes_to_corners_3d(boxes3d)
            rlt.append(corners3d)
        return rlt

    def pred_to_corner(self, preds):

        if not self.encode_angle_by_sincos:
            xt, yt, zt, dxt, dyt, dzt, rt, *cts = torch.split(preds, 1, dim=-1)
        else:
            xt, yt, zt, dxt, dyt, dzt, cost, sint, *cts = torch.split(preds, 1, dim=-1)
            rt = torch.atan2(sint, cost)

        dxg = torch.exp(dxt)
        dyg = torch.exp(dyt)
        dzg = torch.exp(dzt)

        all_boxes = torch.cat([xt, yt, zt, dxg, dyg, dzg], dim=-1)


        rlt = []
        for i in range(len(all_boxes)):
            boxes3d = all_boxes[i]
            corners3d = CenterCoder.boxes_to_corners_3d(boxes3d)
            rlt.append(corners3d)
            rlt.append(corners3d)

        return rlt

    def get_delta(self, gt_boxes, preds, x_grid, y_grid, **kwargs):
        """
        Args:
            gt_boxes: (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]
            gts: (N, 7 + C) [x, y, z, dx, dy, dz, heading or *[cos, sin], ...]

        Returns:

        """
        gt_boxes = CenterCoder._prep(gt_boxes)

        xg, yg, zg, dxg, dyg, dzg, rg, *cgs = torch.split(gt_boxes, 1, dim=-1)  # has been log
        if not self.encode_angle_by_sincos:
            xp, yp, zp, dxp, dyp, dzp, rp, *cps = torch.split(preds, 1, dim=-1)
        else:
            xp, yp, zp, dxp, dyp, dzp, cosp, sinp, *cps = torch.split(preds, 1, dim=-1)

        xt = xg - xp
        yt = yg - yp
        zt = zg - zp

        dxt = torch.log(dxg) - dxp
        dyt = torch.log(dyg) - dyp
        dzt = torch.log(dzg) - dzp

        # rectify: calculate pred center angle
        if self.rectify:
            rel_thetap = torch.atan2(sinp, cosp)
            pred_box_theta = torch.atan2(y_grid, x_grid).to(rel_thetap.device).unsqueeze(-1)

            thetap = rel_thetap + pred_box_theta
            thetap_ind = thetap.new_zeros(tuple(thetap.shape))
            thetap_ind[thetap > np.pi] = -1
            thetap_ind[thetap < -np.pi] = 1
            thetap_ind *= 2 * np.pi
            thetap += thetap_ind

            global_cosp = torch.cos(thetap)
            global_sinp = torch.sin(thetap)

        if self.encode_angle_by_sincos:
            rt_cos = torch.cos(rg) - global_cosp
            rt_sin = torch.sin(rg) - global_sinp
            rts = [rt_cos, rt_sin]
        else:
            raise NotImplementedError

        cts = [g - a for g, a in zip(cgs, cps)]
        return torch.cat([xt, yt, zt, dxt, dyt, dzt, *rts, *cts], dim=-1)

    # no use in this proj
    def decode_torch(self, preds):
        """
        Args:
            box_encodings: (B, N, 7 + C) or (N, 7 + C) [x, y, z, dx, dy, dz, heading or *[cos, sin], ...]
        Returns:

        """
        if not self.encode_angle_by_sincos:
            xt, yt, zt, dxt, dyt, dzt, rt, *cts = torch.split(preds, 1, dim=-1)
        else:
            xt, yt, zt, dxt, dyt, dzt, cost, sint, *cts = torch.split(preds, 1, dim=-1)

        dxg = torch.exp(dxt)
        dyg = torch.exp(dyt)
        dzg = torch.exp(dzt)

        if self.encode_angle_by_sincos:
            rg = torch.atan2(sint, cost)
        else:
            rg = rt * self.period

        cgs = [t for t in cts]
        return torch.cat([xt, yt, zt, dxg, dyg, dzg, rg, *cgs], dim=-1)


    def encode_with_rois(self, boxes, rois):
        """
        Args:
            boxes: (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]
            anchors: (N, 7 + C) [x, y, z, dx, dy, dz, heading or *[cos, sin], ...]

        Returns:

        """
        rois[:, 3:6] = torch.clamp_min(rois[:, 3:6], min=1e-5)
        boxes[:, 3:6] = torch.clamp_min(boxes[:, 3:6], min=1e-5)

        xa, ya, za, dxa, dya, dza, ra, *cas = torch.split(rois, 1, dim=-1)
        xg, yg, zg, dxg, dyg, dzg, rg, *cgs = torch.split(boxes, 1, dim=-1)

        diagonal = torch.sqrt(dxa ** 2 + dya ** 2)
        xt = (xg - xa) / diagonal
        yt = (yg - ya) / diagonal
        zt = (zg - za) / dza
        dxt = torch.log(dxg / dxa)
        dyt = torch.log(dyg / dya)
        dzt = torch.log(dzg / dza)
        if self.encode_angle_by_sincos:
            rt_cos = torch.cos(rg) - torch.cos(ra)
            rt_sin = torch.sin(rg) - torch.sin(ra)
            rts = [rt_cos, rt_sin]
        else:
            rts = [rg - ra]

        cts = [g - a for g, a in zip(cgs, cas)]
        return torch.cat([xt, yt, zt, dxt, dyt, dzt, *rts, *cts], dim=-1)


class CenterCoderPolar(object):
    def __init__(self, code_size=7, encode_angle_by_sincos=False, period=2 * np.pi, rectify=False, **kwargs):
        super().__init__()
        self.code_size = code_size
        self.encode_angle_by_sincos = encode_angle_by_sincos
        self.period = period
        self.rectify = rectify
        if self.encode_angle_by_sincos:
            self.code_size += 1

    @staticmethod
    def _rotate_points_along_z(points, angle):
        cosa = torch.cos(angle)
        sina = torch.sin(angle)
        zeros = angle.new_zeros(points.shape[0])
        ones = angle.new_ones(points.shape[0])
        rot_matrix = torch.stack((
            cosa, sina, zeros,
            -sina, cosa, zeros,
            zeros, zeros, ones
        ), dim=1).view(-1, 3, 3).float()
        points_rot = torch.matmul(points[:, :, 0:3], rot_matrix)
        points_rot = torch.cat((points_rot, points[:, :, 3:]), dim=-1)
        return points_rot

    @staticmethod
    def boxes_to_corners_3d(boxes3d):
        template = boxes3d.new_tensor((
            [1, 1, -1], [1, -1, -1], [-1, -1, -1], [-1, 1, -1],
            [1, 1, 1], [1, -1, 1], [-1, -1, 1], [-1, 1, 1],
        )) / 2

        corners3d = boxes3d[:, None, 3:6].repeat(1, 8, 1) * template[None, :, :]
        corners3d = CenterCoder._rotate_points_along_z(corners3d.view(-1, 8, 3), boxes3d[:, 6]).view(-1, 8, 3)
        corners3d += boxes3d[:, None, 0:3]
        return corners3d

    @staticmethod
    def _prep(input):
        slices = [input[..., :3], input[..., 3:6], input[..., 6:]]
        slices[1] = torch.clamp_min(slices[1], min=1e-5)
        return torch.cat(slices, dim=-1)

    # encode gt box in polar space
    def encode(self, gt_boxes):
        rlt = []
        for i in range(len(gt_boxes)):
            tmp = CenterCoder._prep(gt_boxes[i])

            xg, yg, zg, dxg, dyg, dzg, rg, *cgs = torch.split(tmp, 1, dim=-1)

            # center from cart to polar
            rhog = torch.sqrt(xg ** 2 + yg ** 2)
            phig = torch.atan2(yg, xg)

            radiug = torch.sqrt(dxg ** 2 + dyg ** 2) / 2  # half
            ratiog = dyg / dxg  # dy/dx

            radiun = torch.log(radiug)
            dzn = torch.log(dzg)

            # rectify: calculate gt center angle
            if self.rectify:
                rel_rg = rg - phig
                rel_rg_ind = rel_rg.new_zeros(tuple(rel_rg.shape))
                rel_rg_ind[rel_rg > np.pi] = -1
                rel_rg_ind[rel_rg < -np.pi] = 1
                rel_rg_ind *= 2 * np.pi
                rel_rg += rel_rg_ind
                rg = rel_rg

            if self.encode_angle_by_sincos:
                cosg = torch.cos(rg)
                sing = torch.sin(rg)
                rgs = [cosg, sing]
            else:
                rgs = [rg, ]

            rlt.append(torch.cat([rhog, phig, zg, radiun, dzn, ratiog, *rgs, *cgs], dim=-1))

        return rlt

    def gt_to_corner(self, gt_boxes):
        rlt = []
        for i in range(len(gt_boxes)):
            boxes3d = CenterCoder._prep(gt_boxes[i])
            corners3d = CenterCoder.boxes_to_corners_3d(boxes3d)
            rlt.append(corners3d)
        return rlt

    def pred_to_corner(self, preds):

        if not self.encode_angle_by_sincos:
            xt, yt, zt, dxt, dyt, dzt, rt, *cts = torch.split(preds, 1, dim=-1)
        else:
            xt, yt, zt, dxt, dyt, dzt, cost, sint, *cts = torch.split(preds, 1, dim=-1)
            rt = torch.atan2(sint, cost)

        dxg = torch.exp(dxt)
        dyg = torch.exp(dyt)
        dzg = torch.exp(dzt)

        all_boxes = torch.cat([xt, yt, zt, dxg, dyg, dzg], dim=-1)


        rlt = []
        for i in range(len(all_boxes)):
            boxes3d = all_boxes[i]
            corners3d = CenterCoder.boxes_to_corners_3d(boxes3d)
            rlt.append(corners3d)
            rlt.append(corners3d)

        return rlt

    def get_delta(self, gt_boxes, preds, **kwargs):
        """
        Args:
            gt_boxes: (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]
            gts: (N, 7 + C) [x, y, z, dx, dy, dz, heading or *[cos, sin], ...]

        Returns:

        """
        gt_boxes = CenterCoder._prep(gt_boxes)

        xg, yg, zg, dxg, dyg, dzg, rg, *cgs = torch.split(gt_boxes, 1, dim=-1)  # has been log
        if not self.encode_angle_by_sincos:
            rhop, phip, zp, radiup, dzp, ratiop, headp, *cps = torch.split(preds, 1, dim=-1)
        else:
            rhop, phip, zp, radiup, dzp, ratiop, cosp, sinp, *cps = torch.split(preds, 1, dim=-1)

        # center from cart to polar
        rhog = torch.sqrt(xg ** 2 + yg ** 2)
        phig = torch.atan2(yg, xg)

        rhot = rhog - rhop
        phit = phig - phip

        zt = zg - zp

        # circle box radius and ratio
        radiug = torch.sqrt(dxg ** 2 + dyg ** 2) / 2  # half
        ratiog = dyg / dxg  # dy/dx

        radiut = torch.log(radiug) - radiup  # regress radiu in log
        dzt = torch.log(dzg) - dzp  # regress height in log

        ratiot = ratiog - ratiop  # regress ratio

        # rectify: calculate gt center angle
        if self.rectify:
            rel_rg = rg - phig
            rel_rg_ind = rel_rg.new_zeros(tuple(rel_rg.shape))
            rel_rg_ind[rel_rg > np.pi] = -1
            rel_rg_ind[rel_rg < -np.pi] = 1
            rel_rg_ind *= 2 * np.pi
            rel_rg += rel_rg_ind
            rg = rel_rg

        if self.encode_angle_by_sincos:
            rt_cos = torch.cos(rg) - cosp
            rt_sin = torch.sin(rg) - sinp
            rts = [rt_cos, rt_sin]
        else:
            raise NotImplementedError

        cts = [g - a for g, a in zip(cgs, cps)]
        return torch.cat([rhot, phit, zt, radiut, dzt, ratiot, *rts, *cts], dim=-1)

    # no use in this proj
    def decode_torch(self, preds):
        """
        Args:
            box_encodings: (B, N, 7 + C) or (N, 7 + C) [x, y, z, dx, dy, dz, heading or *[cos, sin], ...]
        Returns:

        """
        if not self.encode_angle_by_sincos:
            xt, yt, zt, dxt, dyt, dzt, rt, *cts = torch.split(preds, 1, dim=-1)
        else:
            xt, yt, zt, dxt, dyt, dzt, cost, sint, *cts = torch.split(preds, 1, dim=-1)

        dxg = torch.exp(dxt)
        dyg = torch.exp(dyt)
        dzg = torch.exp(dzt)

        if self.encode_angle_by_sincos:
            rg = torch.atan2(sint, cost)
        else:
            rg = rt * self.period

        cgs = [t for t in cts]
        return torch.cat([xt, yt, zt, dxg, dyg, dzg, rg, *cgs], dim=-1)


    def encode_with_rois(self, boxes, rois):
        """
        Args:
            boxes: (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]
            anchors: (N, 7 + C) [x, y, z, dx, dy, dz, heading or *[cos, sin], ...]

        Returns:

        """
        rois[:, 3:6] = torch.clamp_min(rois[:, 3:6], min=1e-5)
        boxes[:, 3:6] = torch.clamp_min(boxes[:, 3:6], min=1e-5)

        xa, ya, za, dxa, dya, dza, ra, *cas = torch.split(rois, 1, dim=-1)
        xg, yg, zg, dxg, dyg, dzg, rg, *cgs = torch.split(boxes, 1, dim=-1)

        diagonal = torch.sqrt(dxa ** 2 + dya ** 2)
        xt = (xg - xa) / diagonal
        yt = (yg - ya) / diagonal
        zt = (zg - za) / dza
        dxt = torch.log(dxg / dxa)
        dyt = torch.log(dyg / dya)
        dzt = torch.log(dzg / dza)
        if self.encode_angle_by_sincos:
            rt_cos = torch.cos(rg) - torch.cos(ra)
            rt_sin = torch.sin(rg) - torch.sin(ra)
            rts = [rt_cos, rt_sin]
        else:
            rts = [rg - ra]

        cts = [g - a for g, a in zip(cgs, cas)]
        return torch.cat([xt, yt, zt, dxt, dyt, dzt, *rts, *cts], dim=-1)


class CenterCoderPolarTP(object):
    def __init__(self, code_size=7, encode_angle_by_sincos=False, period=2 * np.pi, rectify=False, template_ratio=None, **kwargs):
        super().__init__()
        self.code_size = code_size
        self.encode_angle_by_sincos = encode_angle_by_sincos
        self.period = period
        self.rectify = rectify
        self.template_ratio = template_ratio
        assert self.template_ratio is not None  # ratio = dy / dx
        if self.encode_angle_by_sincos:
            self.code_size += 1

    @staticmethod
    def _rotate_points_along_z(points, angle):
        cosa = torch.cos(angle)
        sina = torch.sin(angle)
        zeros = angle.new_zeros(points.shape[0])
        ones = angle.new_ones(points.shape[0])
        rot_matrix = torch.stack((
            cosa, sina, zeros,
            -sina, cosa, zeros,
            zeros, zeros, ones
        ), dim=1).view(-1, 3, 3).float()
        points_rot = torch.matmul(points[:, :, 0:3], rot_matrix)
        points_rot = torch.cat((points_rot, points[:, :, 3:]), dim=-1)
        return points_rot

    @staticmethod
    def boxes_to_corners_3d(boxes3d):
        template = boxes3d.new_tensor((
            [1, 1, -1], [1, -1, -1], [-1, -1, -1], [-1, 1, -1],
            [1, 1, 1], [1, -1, 1], [-1, -1, 1], [-1, 1, 1],
        )) / 2

        corners3d = boxes3d[:, None, 3:6].repeat(1, 8, 1) * template[None, :, :]
        corners3d = CenterCoder._rotate_points_along_z(corners3d.view(-1, 8, 3), boxes3d[:, 6]).view(-1, 8, 3)
        corners3d += boxes3d[:, None, 0:3]
        return corners3d

    @staticmethod
    def _prep(input):
        slices = [input[..., :3], input[..., 3:6], input[..., 6:]]
        slices[1] = torch.clamp_min(slices[1], min=1e-5)
        return torch.cat(slices, dim=-1)

    # encode gt box in polar space
    def encode(self, gt_boxes):
        rlt = []
        for i in range(len(gt_boxes)):
            tmp = CenterCoder._prep(gt_boxes[i])

            xg, yg, zg, dxg, dyg, dzg, rg, *cgs = torch.split(tmp, 1, dim=-1)

            # center from cart to polar
            rhog = torch.sqrt(xg ** 2 + yg ** 2)
            phig = torch.atan2(yg, xg)

            radiug = torch.sqrt(dxg ** 2 + dyg ** 2) / 2  # half
            ratiog = dyg / dxg  # dy/dx
            ratiog = ratiog - self.template_ratio  # template ratio

            radiun = torch.log(radiug)
            dzn = torch.log(dzg)

            # rectify: calculate gt center angle
            if self.rectify:
                rel_rg = rg - phig
                rel_rg_ind = rel_rg.new_zeros(tuple(rel_rg.shape))
                rel_rg_ind[rel_rg > np.pi] = -1
                rel_rg_ind[rel_rg < -np.pi] = 1
                rel_rg_ind *= 2 * np.pi
                rel_rg += rel_rg_ind
                rg = rel_rg

            if self.encode_angle_by_sincos:
                cosg = torch.cos(rg)
                sing = torch.sin(rg)
                rgs = [cosg, sing]
            else:
                rgs = [rg, ]

            rlt.append(torch.cat([rhog, phig, zg, radiun, dzn, ratiog, *rgs, *cgs], dim=-1))

        return rlt

    def gt_to_corner(self, gt_boxes):
        rlt = []
        for i in range(len(gt_boxes)):
            boxes3d = CenterCoder._prep(gt_boxes[i])
            corners3d = CenterCoder.boxes_to_corners_3d(boxes3d)
            rlt.append(corners3d)
        return rlt

    def pred_to_corner(self, preds):

        if not self.encode_angle_by_sincos:
            xt, yt, zt, dxt, dyt, dzt, rt, *cts = torch.split(preds, 1, dim=-1)
        else:
            xt, yt, zt, dxt, dyt, dzt, cost, sint, *cts = torch.split(preds, 1, dim=-1)
            rt = torch.atan2(sint, cost)

        dxg = torch.exp(dxt)
        dyg = torch.exp(dyt)
        dzg = torch.exp(dzt)

        all_boxes = torch.cat([xt, yt, zt, dxg, dyg, dzg], dim=-1)


        rlt = []
        for i in range(len(all_boxes)):
            boxes3d = all_boxes[i]
            corners3d = CenterCoder.boxes_to_corners_3d(boxes3d)
            rlt.append(corners3d)
            rlt.append(corners3d)

        return rlt

    def get_delta(self, gt_boxes, preds, **kwargs):
        """
        Args:
            gt_boxes: (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]
            gts: (N, 7 + C) [x, y, z, dx, dy, dz, heading or *[cos, sin], ...]

        Returns:

        """
        gt_boxes = CenterCoder._prep(gt_boxes)

        xg, yg, zg, dxg, dyg, dzg, rg, *cgs = torch.split(gt_boxes, 1, dim=-1)  # has been log
        if not self.encode_angle_by_sincos:
            rhop, phip, zp, radiup, dzp, ratiop, headp, *cps = torch.split(preds, 1, dim=-1)
        else:
            rhop, phip, zp, radiup, dzp, ratiop, cosp, sinp, *cps = torch.split(preds, 1, dim=-1)

        # center from cart to polar
        rhog = torch.sqrt(xg ** 2 + yg ** 2)
        phig = torch.atan2(yg, xg)

        rhot = rhog - rhop
        phit = phig - phip

        zt = zg - zp

        # circle box radius and ratio
        radiug = torch.sqrt(dxg ** 2 + dyg ** 2) / 2  # half
        ratiog = dyg / dxg  # dy/dx

        radiut = torch.log(radiug) - radiup  # regress radiu in log
        dzt = torch.log(dzg) - dzp  # regress height in log

        ratiot = ratiog - self.template_ratio - ratiop  # regress ratio, template ratio

        # rectify: calculate gt center angle
        if self.rectify:
            rel_rg = rg - phig
            rel_rg_ind = rel_rg.new_zeros(tuple(rel_rg.shape))
            rel_rg_ind[rel_rg > np.pi] = -1
            rel_rg_ind[rel_rg < -np.pi] = 1
            rel_rg_ind *= 2 * np.pi
            rel_rg += rel_rg_ind
            rg = rel_rg

        if self.encode_angle_by_sincos:
            rt_cos = torch.cos(rg) - cosp
            rt_sin = torch.sin(rg) - sinp
            rts = [rt_cos, rt_sin]
        else:
            raise NotImplementedError

        cts = [g - a for g, a in zip(cgs, cps)]
        return torch.cat([rhot, phit, zt, radiut, dzt, ratiot, *rts, *cts], dim=-1)

    # no use in this proj
    def decode_torch(self, preds):
        """
        Args:
            box_encodings: (B, N, 7 + C) or (N, 7 + C) [x, y, z, dx, dy, dz, heading or *[cos, sin], ...]
        Returns:

        """
        if not self.encode_angle_by_sincos:
            xt, yt, zt, dxt, dyt, dzt, rt, *cts = torch.split(preds, 1, dim=-1)
        else:
            xt, yt, zt, dxt, dyt, dzt, cost, sint, *cts = torch.split(preds, 1, dim=-1)

        dxg = torch.exp(dxt)
        dyg = torch.exp(dyt)
        dzg = torch.exp(dzt)

        if self.encode_angle_by_sincos:
            rg = torch.atan2(sint, cost)
        else:
            rg = rt * self.period

        cgs = [t for t in cts]
        return torch.cat([xt, yt, zt, dxg, dyg, dzg, rg, *cgs], dim=-1)


    def encode_with_rois(self, boxes, rois):
        """
        Args:
            boxes: (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]
            anchors: (N, 7 + C) [x, y, z, dx, dy, dz, heading or *[cos, sin], ...]

        Returns:

        """
        rois[:, 3:6] = torch.clamp_min(rois[:, 3:6], min=1e-5)
        boxes[:, 3:6] = torch.clamp_min(boxes[:, 3:6], min=1e-5)

        xa, ya, za, dxa, dya, dza, ra, *cas = torch.split(rois, 1, dim=-1)
        xg, yg, zg, dxg, dyg, dzg, rg, *cgs = torch.split(boxes, 1, dim=-1)

        diagonal = torch.sqrt(dxa ** 2 + dya ** 2)
        xt = (xg - xa) / diagonal
        yt = (yg - ya) / diagonal
        zt = (zg - za) / dza
        dxt = torch.log(dxg / dxa)
        dyt = torch.log(dyg / dya)
        dzt = torch.log(dzg / dza)
        if self.encode_angle_by_sincos:
            rt_cos = torch.cos(rg) - torch.cos(ra)
            rt_sin = torch.sin(rg) - torch.sin(ra)
            rts = [rt_cos, rt_sin]
        else:
            rts = [rg - ra]

        cts = [g - a for g, a in zip(cgs, cas)]
        return torch.cat([xt, yt, zt, dxt, dyt, dzt, *rts, *cts], dim=-1)


class CenterCoderPolarSIG(object):
    def __init__(self, code_size=7, encode_angle_by_sincos=False, period=2 * np.pi, rectify=False, **kwargs):
        super().__init__()
        self.code_size = code_size
        self.encode_angle_by_sincos = encode_angle_by_sincos
        self.period = period
        self.rectify = rectify
        if self.encode_angle_by_sincos:
            self.code_size += 1

    @staticmethod
    def _rotate_points_along_z(points, angle):
        cosa = torch.cos(angle)
        sina = torch.sin(angle)
        zeros = angle.new_zeros(points.shape[0])
        ones = angle.new_ones(points.shape[0])
        rot_matrix = torch.stack((
            cosa, sina, zeros,
            -sina, cosa, zeros,
            zeros, zeros, ones
        ), dim=1).view(-1, 3, 3).float()
        points_rot = torch.matmul(points[:, :, 0:3], rot_matrix)
        points_rot = torch.cat((points_rot, points[:, :, 3:]), dim=-1)
        return points_rot

    @staticmethod
    def boxes_to_corners_3d(boxes3d):
        template = boxes3d.new_tensor((
            [1, 1, -1], [1, -1, -1], [-1, -1, -1], [-1, 1, -1],
            [1, 1, 1], [1, -1, 1], [-1, -1, 1], [-1, 1, 1],
        )) / 2

        corners3d = boxes3d[:, None, 3:6].repeat(1, 8, 1) * template[None, :, :]
        corners3d = CenterCoder._rotate_points_along_z(corners3d.view(-1, 8, 3), boxes3d[:, 6]).view(-1, 8, 3)
        corners3d += boxes3d[:, None, 0:3]
        return corners3d

    @staticmethod
    def _prep(input):
        slices = [input[..., :3], input[..., 3:6], input[..., 6:]]
        slices[1] = torch.clamp_min(slices[1], min=1e-5)
        return torch.cat(slices, dim=-1)

    # encode gt box in polar space
    def encode(self, gt_boxes):
        rlt = []
        for i in range(len(gt_boxes)):
            tmp = CenterCoder._prep(gt_boxes[i])

            xg, yg, zg, dxg, dyg, dzg, rg, *cgs = torch.split(tmp, 1, dim=-1)

            # center from cart to polar
            rhog = torch.sqrt(xg ** 2 + yg ** 2)
            phig = torch.atan2(yg, xg)

            radiug = torch.sqrt(dxg ** 2 + dyg ** 2) / 2  # half
            ratiog = dyg / dxg  # dy/dx, > 1

            ratiog = torch.logit(1 / ratiog)  # trans to 1 / r and logistic on ratio

            radiun = torch.log(radiug)
            dzn = torch.log(dzg)

            # rectify: calculate gt center angle
            if self.rectify:
                rel_rg = rg - phig
                rel_rg_ind = rel_rg.new_zeros(tuple(rel_rg.shape))
                rel_rg_ind[rel_rg > np.pi] = -1
                rel_rg_ind[rel_rg < -np.pi] = 1
                rel_rg_ind *= 2 * np.pi
                rel_rg += rel_rg_ind
                rg = rel_rg

            if self.encode_angle_by_sincos:
                cosg = torch.cos(rg)
                sing = torch.sin(rg)
                rgs = [cosg, sing]
            else:
                rgs = [rg, ]

            rlt.append(torch.cat([rhog, phig, zg, radiun, dzn, ratiog, *rgs, *cgs], dim=-1))

        return rlt

    def gt_to_corner(self, gt_boxes):
        rlt = []
        for i in range(len(gt_boxes)):
            boxes3d = CenterCoder._prep(gt_boxes[i])
            corners3d = CenterCoder.boxes_to_corners_3d(boxes3d)
            rlt.append(corners3d)
        return rlt

    def pred_to_corner(self, preds):

        if not self.encode_angle_by_sincos:
            xt, yt, zt, dxt, dyt, dzt, rt, *cts = torch.split(preds, 1, dim=-1)
        else:
            xt, yt, zt, dxt, dyt, dzt, cost, sint, *cts = torch.split(preds, 1, dim=-1)
            rt = torch.atan2(sint, cost)

        dxg = torch.exp(dxt)
        dyg = torch.exp(dyt)
        dzg = torch.exp(dzt)

        all_boxes = torch.cat([xt, yt, zt, dxg, dyg, dzg], dim=-1)


        rlt = []
        for i in range(len(all_boxes)):
            boxes3d = all_boxes[i]
            corners3d = CenterCoder.boxes_to_corners_3d(boxes3d)
            rlt.append(corners3d)
            rlt.append(corners3d)

        return rlt

    def get_delta(self, gt_boxes, preds, **kwargs):
        """
        Args:
            gt_boxes: (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]
            gts: (N, 7 + C) [x, y, z, dx, dy, dz, heading or *[cos, sin], ...]

        Returns:

        """
        gt_boxes = CenterCoder._prep(gt_boxes)

        xg, yg, zg, dxg, dyg, dzg, rg, *cgs = torch.split(gt_boxes, 1, dim=-1)  # has been log
        if not self.encode_angle_by_sincos:
            rhop, phip, zp, radiup, dzp, ratiop, headp, *cps = torch.split(preds, 1, dim=-1)
        else:
            rhop, phip, zp, radiup, dzp, ratiop, cosp, sinp, *cps = torch.split(preds, 1, dim=-1)

        # center from cart to polar
        rhog = torch.sqrt(xg ** 2 + yg ** 2)
        phig = torch.atan2(yg, xg)

        rhot = rhog - rhop
        phit = phig - phip

        zt = zg - zp

        # circle box radius and ratio
        radiug = torch.sqrt(dxg ** 2 + dyg ** 2) / 2  # half
        ratiog = dyg / dxg  # dy/dx

        radiut = torch.log(radiug) - radiup  # regress radiu in log
        dzt = torch.log(dzg) - dzp  # regress height in log

        ratiop = 1 / torch.sigmoid(ratiop)  # sigmoid on ratio, and trans to 1 / r
        ratiot = ratiog - ratiop  # regress ratio

        # rectify: calculate gt center angle
        if self.rectify:
            rel_rg = rg - phig
            rel_rg_ind = rel_rg.new_zeros(tuple(rel_rg.shape))
            rel_rg_ind[rel_rg > np.pi] = -1
            rel_rg_ind[rel_rg < -np.pi] = 1
            rel_rg_ind *= 2 * np.pi
            rel_rg += rel_rg_ind
            rg = rel_rg

        if self.encode_angle_by_sincos:
            rt_cos = torch.cos(rg) - cosp
            rt_sin = torch.sin(rg) - sinp
            rts = [rt_cos, rt_sin]
        else:
            raise NotImplementedError

        cts = [g - a for g, a in zip(cgs, cps)]
        return torch.cat([rhot, phit, zt, radiut, dzt, ratiot, *rts, *cts], dim=-1)

    # no use in this proj
    def decode_torch(self, preds):
        """
        Args:
            box_encodings: (B, N, 7 + C) or (N, 7 + C) [x, y, z, dx, dy, dz, heading or *[cos, sin], ...]
        Returns:

        """
        if not self.encode_angle_by_sincos:
            xt, yt, zt, dxt, dyt, dzt, rt, *cts = torch.split(preds, 1, dim=-1)
        else:
            xt, yt, zt, dxt, dyt, dzt, cost, sint, *cts = torch.split(preds, 1, dim=-1)

        dxg = torch.exp(dxt)
        dyg = torch.exp(dyt)
        dzg = torch.exp(dzt)

        if self.encode_angle_by_sincos:
            rg = torch.atan2(sint, cost)
        else:
            rg = rt * self.period

        cgs = [t for t in cts]
        return torch.cat([xt, yt, zt, dxg, dyg, dzg, rg, *cgs], dim=-1)


    def encode_with_rois(self, boxes, rois):
        """
        Args:
            boxes: (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]
            anchors: (N, 7 + C) [x, y, z, dx, dy, dz, heading or *[cos, sin], ...]

        Returns:

        """
        rois[:, 3:6] = torch.clamp_min(rois[:, 3:6], min=1e-5)
        boxes[:, 3:6] = torch.clamp_min(boxes[:, 3:6], min=1e-5)

        xa, ya, za, dxa, dya, dza, ra, *cas = torch.split(rois, 1, dim=-1)
        xg, yg, zg, dxg, dyg, dzg, rg, *cgs = torch.split(boxes, 1, dim=-1)

        diagonal = torch.sqrt(dxa ** 2 + dya ** 2)
        xt = (xg - xa) / diagonal
        yt = (yg - ya) / diagonal
        zt = (zg - za) / dza
        dxt = torch.log(dxg / dxa)
        dyt = torch.log(dyg / dya)
        dzt = torch.log(dzg / dza)
        if self.encode_angle_by_sincos:
            rt_cos = torch.cos(rg) - torch.cos(ra)
            rt_sin = torch.sin(rg) - torch.sin(ra)
            rts = [rt_cos, rt_sin]
        else:
            rts = [rg - ra]

        cts = [g - a for g, a in zip(cgs, cas)]
        return torch.cat([xt, yt, zt, dxt, dyt, dzt, *rts, *cts], dim=-1)


# useless
class CenterCoderTP(object):
    def __init__(self, code_size=7, encode_angle_by_sincos=False, period=2 * np.pi, rectify=False, **kwargs):
        super().__init__()
        self.code_size = code_size
        self.encode_angle_by_sincos = encode_angle_by_sincos
        self.period = period
        self.rectify = rectify
        if self.encode_angle_by_sincos:
            self.code_size += 1

    @staticmethod
    def _rotate_points_along_z(points, angle):
        cosa = torch.cos(angle)
        sina = torch.sin(angle)
        zeros = angle.new_zeros(points.shape[0])
        ones = angle.new_ones(points.shape[0])
        rot_matrix = torch.stack((
            cosa, sina, zeros,
            -sina, cosa, zeros,
            zeros, zeros, ones
        ), dim=1).view(-1, 3, 3).float()
        points_rot = torch.matmul(points[:, :, 0:3], rot_matrix)
        points_rot = torch.cat((points_rot, points[:, :, 3:]), dim=-1)
        return points_rot

    @staticmethod
    def boxes_to_corners_3d(boxes3d):
        template = boxes3d.new_tensor((
            [1, 1, -1], [1, -1, -1], [-1, -1, -1], [-1, 1, -1],
            [1, 1, 1], [1, -1, 1], [-1, -1, 1], [-1, 1, 1],
        )) / 2

        corners3d = boxes3d[:, None, 3:6].repeat(1, 8, 1) * template[None, :, :]
        corners3d = CenterCoder._rotate_points_along_z(corners3d.view(-1, 8, 3), boxes3d[:, 6]).view(-1, 8, 3)
        corners3d += boxes3d[:, None, 0:3]
        return corners3d

    @staticmethod
    def _prep(input):
        slices = [input[..., :3], input[..., 3:6], input[..., 6:]]
        slices[1] = torch.clamp_min(slices[1], min=1e-5)
        return torch.cat(slices, dim=-1)

    def encode(self, gt_boxes, templates_box):
        rlt = []
        for i in range(len(gt_boxes)):
            tmp = CenterCoder._prep(gt_boxes[i])

            xg, yg, zg, dxg, dyg, dzg, rg, *cgs = torch.split(tmp, 1, dim=-1)

            dxn = dxg - templates_box[0]
            dyn = dyg - templates_box[1]
            dzn = dzg - templates_box[2]

            # rectify: calculate gt center angle
            if self.rectify:
                phig = torch.atan2(yg, xg)
                rel_rg = rg - phig
                rel_rg_ind = rel_rg.new_zeros(tuple(rel_rg.shape))
                rel_rg_ind[rel_rg > np.pi] = -1
                rel_rg_ind[rel_rg < -np.pi] = 1
                rel_rg_ind *= 2 * np.pi
                rel_rg += rel_rg_ind
                rg = rel_rg

            if self.encode_angle_by_sincos:
                cosg = torch.cos(rg)
                sing = torch.sin(rg)
                rgs = [cosg, sing]
            else:
                rgs = [rg, ]

            rlt.append(torch.cat([xg, yg, zg, dxn, dyn, dzn, *rgs, *cgs], dim=-1))

        return rlt

    def gt_to_corner(self, gt_boxes):
        rlt = []
        for i in range(len(gt_boxes)):
            boxes3d = CenterCoder._prep(gt_boxes[i])
            corners3d = CenterCoder.boxes_to_corners_3d(boxes3d)
            rlt.append(corners3d)
        return rlt

    def pred_to_corner(self, preds):

        if not self.encode_angle_by_sincos:
            xt, yt, zt, dxt, dyt, dzt, rt, *cts = torch.split(preds, 1, dim=-1)
        else:
            xt, yt, zt, dxt, dyt, dzt, cost, sint, *cts = torch.split(preds, 1, dim=-1)
            rt = torch.atan2(sint, cost)

        dxg = torch.exp(dxt)
        dyg = torch.exp(dyt)
        dzg = torch.exp(dzt)

        all_boxes = torch.cat([xt, yt, zt, dxg, dyg, dzg], dim=-1)


        rlt = []
        for i in range(len(all_boxes)):
            boxes3d = all_boxes[i]
            corners3d = CenterCoder.boxes_to_corners_3d(boxes3d)
            rlt.append(corners3d)
            rlt.append(corners3d)

        return rlt

    def get_delta(self, gt_boxes, preds, template_box, **kwargs):
        """
        Args:
            gt_boxes: (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]
            gts: (N, 7 + C) [x, y, z, dx, dy, dz, heading or *[cos, sin], ...]

        Returns:

        """
        gt_boxes = CenterCoder._prep(gt_boxes)

        xg, yg, zg, dxg, dyg, dzg, rg, *cgs = torch.split(gt_boxes, 1, dim=-1)  # has been log
        if not self.encode_angle_by_sincos:
            xp, yp, zp, dxp, dyp, dzp, rp, *cps = torch.split(preds, 1, dim=-1)
        else:
            xp, yp, zp, dxp, dyp, dzp, cosp, sinp, *cps = torch.split(preds, 1, dim=-1)

        xt = xg - xp
        yt = yg - yp
        zt = zg - zp

        dxt = dxg - dxp - template_box[0]
        dyt = dyg - dyp - template_box[1]
        dzt = dzg - dzp - template_box[2]

        # rectify: calculate gt center angle
        if self.rectify:
            phig = torch.atan2(yg, xg)
            rel_rg = rg - phig
            rel_rg_ind = rel_rg.new_zeros(tuple(rel_rg.shape))
            rel_rg_ind[rel_rg > np.pi] = -1
            rel_rg_ind[rel_rg < -np.pi] = 1
            rel_rg_ind *= 2 * np.pi
            rel_rg += rel_rg_ind
            rg = rel_rg

        if self.encode_angle_by_sincos:
            rt_cos = torch.cos(rg) - cosp
            rt_sin = torch.sin(rg) - sinp
            rts = [rt_cos, rt_sin]
        else:
            rts = [(rg / self.period) - rp]

        cts = [g - a for g, a in zip(cgs, cps)]
        return torch.cat([xt, yt, zt, dxt, dyt, dzt, *rts, *cts], dim=-1)

    # no use in this proj
    def decode_torch(self, preds, template_box):
        """
        Args:
            box_encodings: (B, N, 7 + C) or (N, 7 + C) [x, y, z, dx, dy, dz, heading or *[cos, sin], ...]
        Returns:

        """
        if not self.encode_angle_by_sincos:
            xt, yt, zt, dxt, dyt, dzt, rt, *cts = torch.split(preds, 1, dim=-1)
        else:
            xt, yt, zt, dxt, dyt, dzt, cost, sint, *cts = torch.split(preds, 1, dim=-1)

        dxg = dxt + template_box[0]
        dyg = dyt + template_box[1]
        dzg = dzt + template_box[2]

        if self.encode_angle_by_sincos:
            rg = torch.atan2(sint, cost)
        else:
            rg = rt * self.period

        cgs = [t for t in cts]
        return torch.cat([xt, yt, zt, dxg, dyg, dzg, rg, *cgs], dim=-1)


    def encode_with_rois(self, boxes, rois):
        """
        Args:
            boxes: (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]
            anchors: (N, 7 + C) [x, y, z, dx, dy, dz, heading or *[cos, sin], ...]

        Returns:

        """
        rois[:, 3:6] = torch.clamp_min(rois[:, 3:6], min=1e-5)
        boxes[:, 3:6] = torch.clamp_min(boxes[:, 3:6], min=1e-5)

        xa, ya, za, dxa, dya, dza, ra, *cas = torch.split(rois, 1, dim=-1)
        xg, yg, zg, dxg, dyg, dzg, rg, *cgs = torch.split(boxes, 1, dim=-1)

        diagonal = torch.sqrt(dxa ** 2 + dya ** 2)
        xt = (xg - xa) / diagonal
        yt = (yg - ya) / diagonal
        zt = (zg - za) / dza
        dxt = torch.log(dxg / dxa)
        dyt = torch.log(dyg / dya)
        dzt = torch.log(dzg / dza)
        if self.encode_angle_by_sincos:
            rt_cos = torch.cos(rg) - torch.cos(ra)
            rt_sin = torch.sin(rg) - torch.sin(ra)
            rts = [rt_cos, rt_sin]
        else:
            rts = [rg - ra]

        cts = [g - a for g, a in zip(cgs, cas)]
        return torch.cat([xt, yt, zt, dxt, dyt, dzt, *rts, *cts], dim=-1)


class CenterCoderMTTP(object):
    def __init__(self, code_size=7, encode_angle_by_sincos=False, period=2 * np.pi, tmp_box=None, **kwargs):
        super().__init__()
        self.code_size = code_size
        self.encode_angle_by_sincos = encode_angle_by_sincos
        self.period = period
        self.tmp_box = torch.from_numpy(np.array(tmp_box)).cuda().float()
        if self.encode_angle_by_sincos:
            self.code_size += 1

    @staticmethod
    def _rotate_points_along_z(points, angle):
        cosa = torch.cos(angle)
        sina = torch.sin(angle)
        zeros = angle.new_zeros(points.shape[0])
        ones = angle.new_ones(points.shape[0])
        rot_matrix = torch.stack((
            cosa, sina, zeros,
            -sina, cosa, zeros,
            zeros, zeros, ones
        ), dim=1).view(-1, 3, 3).float()
        points_rot = torch.matmul(points[:, :, 0:3], rot_matrix)
        points_rot = torch.cat((points_rot, points[:, :, 3:]), dim=-1)
        return points_rot

    @staticmethod
    def boxes_to_corners_3d(boxes3d):
        template = boxes3d.new_tensor((
            [1, 1, -1], [1, -1, -1], [-1, -1, -1], [-1, 1, -1],
            [1, 1, 1], [1, -1, 1], [-1, -1, 1], [-1, 1, 1],
        )) / 2

        corners3d = boxes3d[:, None, 3:6].repeat(1, 8, 1) * template[None, :, :]
        corners3d = CenterCoder._rotate_points_along_z(corners3d.view(-1, 8, 3), boxes3d[:, 6]).view(-1, 8, 3)
        corners3d += boxes3d[:, None, 0:3]
        return corners3d

    @staticmethod
    def _prep(input):
        slices = [input[..., :3], input[..., 3:6], input[..., 6:]]
        slices[1] = torch.clamp_min(slices[1], min=1e-5)
        return torch.cat(slices, dim=-1)

    def encode(self, gt_boxes, task_id):
        rlt = []
        for i in range(len(gt_boxes)):
            tmp = CenterCoder._prep(gt_boxes[i])

            xg, yg, zg, dxg, dyg, dzg, rg, *cgs = torch.split(tmp, 1, dim=-1)

            point_anchor_size = self.tmp_box[task_id]
            dxa, dya, dza = torch.split(point_anchor_size, 1, dim=-1)
            dxn = torch.log(dxg / dxa)
            dyn = torch.log(dyg / dya)
            dzn = torch.log(dzg / dza)

            if self.encode_angle_by_sincos:
                cosg = torch.cos(rg)
                sing = torch.sin(rg)
                rgs = [cosg, sing]
            else:
                rgs = [rg, ]

            rlt.append(torch.cat([xg, yg, zg, dxn, dyn, dzn, *rgs, *cgs], dim=-1))

        return rlt

    def gt_to_corner(self, gt_boxes):
        rlt = []
        for i in range(len(gt_boxes)):
            boxes3d = CenterCoder._prep(gt_boxes[i])
            corners3d = CenterCoder.boxes_to_corners_3d(boxes3d)
            rlt.append(corners3d)
        return rlt

    def pred_to_corner(self, preds):

        if not self.encode_angle_by_sincos:
            xt, yt, zt, dxt, dyt, dzt, rt, *cts = torch.split(preds, 1, dim=-1)
        else:
            xt, yt, zt, dxt, dyt, dzt, cost, sint, *cts = torch.split(preds, 1, dim=-1)
            rt = torch.atan2(sint, cost)

        dxg = torch.exp(dxt)
        dyg = torch.exp(dyt)
        dzg = torch.exp(dzt)

        all_boxes = torch.cat([xt, yt, zt, dxg, dyg, dzg], dim=-1)


        rlt = []
        for i in range(len(all_boxes)):
            boxes3d = all_boxes[i]
            corners3d = CenterCoder.boxes_to_corners_3d(boxes3d)
            rlt.append(corners3d)
            rlt.append(corners3d)

        return rlt

    def get_delta(self, gt_boxes, preds, task_id, **kwargs):
        """
        Args:
            gt_boxes: (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]
            gts: (N, 7 + C) [x, y, z, dx, dy, dz, heading or *[cos, sin], ...]

        Returns:

        """
        gt_boxes = CenterCoder._prep(gt_boxes)

        xg, yg, zg, dxg, dyg, dzg, rg, *cgs = torch.split(gt_boxes, 1, dim=-1)
        if not self.encode_angle_by_sincos:
            xp, yp, zp, dxp, dyp, dzp, rp, *cps = torch.split(preds, 1, dim=-1)
        else:
            xp, yp, zp, dxp, dyp, dzp, cosp, sinp, *cps = torch.split(preds, 1, dim=-1)

        xt = xg - xp
        yt = yg - yp
        zt = zg - zp

        point_anchor_size = self.tmp_box[task_id]
        dxa, dya, dza = torch.split(point_anchor_size, 1, dim=-1)
        dxt = torch.log(dxg / dxa) - dxp
        dyt = torch.log(dyg / dya) - dyp
        dzt = torch.log(dzg / dza) - dzp

        if self.encode_angle_by_sincos:
            rt_cos = torch.cos(rg) - cosp
            rt_sin = torch.sin(rg) - sinp
            rts = [rt_cos, rt_sin]
        else:
            rts = [(rg / self.period) - rp]

        cts = [g - a for g, a in zip(cgs, cps)]
        return torch.cat([xt, yt, zt, dxt, dyt, dzt, *rts, *cts], dim=-1)

    def decode_torch(self, preds, task_id):
        """
        Args:
            box_encodings: (B, N, 7 + C) or (N, 7 + C) [x, y, z, dx, dy, dz, heading or *[cos, sin], ...]
        Returns:

        """
        if not self.encode_angle_by_sincos:
            xt, yt, zt, dxt, dyt, dzt, rt, *cts = torch.split(preds, 1, dim=-1)
        else:
            xt, yt, zt, dxt, dyt, dzt, cost, sint, *cts = torch.split(preds, 1, dim=-1)

        point_anchor_size = self.tmp_box[task_id]
        dxa, dya, dza = torch.split(point_anchor_size, 1, dim=-1)
        dxg = torch.exp(dxt) * dxa
        dyg = torch.exp(dyt) * dya
        dzg = torch.exp(dzt) * dza

        if self.encode_angle_by_sincos:
            rg = torch.atan2(sint, cost)
        else:
            rg = rt * self.period

        cgs = [t for t in cts]
        return torch.cat([xt, yt, zt, dxg, dyg, dzg, rg, *cgs], dim=-1)


    def encode_with_rois(self, boxes, rois):
        """
        Args:
            boxes: (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]
            anchors: (N, 7 + C) [x, y, z, dx, dy, dz, heading or *[cos, sin], ...]

        Returns:

        """
        rois[:, 3:6] = torch.clamp_min(rois[:, 3:6], min=1e-5)
        boxes[:, 3:6] = torch.clamp_min(boxes[:, 3:6], min=1e-5)

        xa, ya, za, dxa, dya, dza, ra, *cas = torch.split(rois, 1, dim=-1)
        xg, yg, zg, dxg, dyg, dzg, rg, *cgs = torch.split(boxes, 1, dim=-1)

        diagonal = torch.sqrt(dxa ** 2 + dya ** 2)
        xt = (xg - xa) / diagonal
        yt = (yg - ya) / diagonal
        zt = (zg - za) / dza
        dxt = torch.log(dxg / dxa)
        dyt = torch.log(dyg / dya)
        dzt = torch.log(dzg / dza)
        if self.encode_angle_by_sincos:
            rt_cos = torch.cos(rg) - torch.cos(ra)
            rt_sin = torch.sin(rg) - torch.sin(ra)
            rts = [rt_cos, rt_sin]
        else:
            rts = [rg - ra]

        cts = [g - a for g, a in zip(cgs, cas)]
        return torch.cat([xt, yt, zt, dxt, dyt, dzt, *rts, *cts], dim=-1)
