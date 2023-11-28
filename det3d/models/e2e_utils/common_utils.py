import logging
import os
import pickle
import random
import shutil
import subprocess

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp


class GELU(nn.Module):
    def __init__(self):
        super().__init__()
        self.f = nn.functional.gelu

    def forward(self, x):
        return self.f(x)


def check_numpy_to_torch(x):
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).float(), True
    return x, False


def limit_period_torch(val, offset=0, period=2 * np.pi):
    return val - torch.floor(val / period + offset) * period


def limit_period(val, offset=0.5, period=np.pi):
    val, is_numpy = check_numpy_to_torch(val)
    ans = val - torch.floor(val / period + offset) * period
    return ans.numpy() if is_numpy else ans


def drop_info_with_name(info, name):
    ret_info = {}
    keep_indices = [i for i, x in enumerate(info['name']) if x != name]
    for key in info.keys():
        ret_info[key] = info[key][keep_indices]
    return ret_info


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


def mask_points_by_range(points, limit_range):
    mask = (points[:, 0] >= limit_range[0]) & (points[:, 0] <= limit_range[3]) \
           & (points[:, 1] >= limit_range[1]) & (points[:, 1] <= limit_range[4])
    return mask


def get_voxel_centers(voxel_coords, downsample_times, voxel_size, point_cloud_range):
    """
    Args:
        voxel_coords: (N, 3)
        downsample_times:
        voxel_size:
        point_cloud_range:

    Returns:

    """
    assert voxel_coords.shape[1] == 3
    voxel_centers = voxel_coords[:, [2, 1, 0]].float()  # (xyz)
    voxel_size = torch.tensor(voxel_size, device=voxel_centers.device).float() * downsample_times
    pc_range = torch.tensor(point_cloud_range[0:3], device=voxel_centers.device).float()
    voxel_centers = (voxel_centers + 0.5) * voxel_size + pc_range
    return voxel_centers


def create_logger(log_file=None, rank=0, log_level=logging.INFO):
    logger = logging.getLogger(__name__)
    logger.setLevel(log_level if rank == 0 else 'ERROR')
    formatter = logging.Formatter('%(asctime)s  %(levelname)5s  %(message)s')
    console = logging.StreamHandler()
    console.setLevel(log_level if rank == 0 else 'ERROR')
    console.setFormatter(formatter)
    logger.addHandler(console)
    if log_file is not None:
        file_handler = logging.FileHandler(filename=log_file)
        file_handler.setLevel(log_level if rank == 0 else 'ERROR')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def keep_arrays_by_name(gt_names, used_classes):
    inds = [i for i, x in enumerate(gt_names) if x in used_classes]
    inds = np.array(inds, dtype=np.int64)
    return inds


def init_dist_slurm(tcp_port, local_rank, backend='nccl'):
    """
    modified from https://github.com/open-mmlab/mmdetection
    Args:
        tcp_port:
        backend:

    Returns:

    """
    proc_id = int(os.environ['SLURM_PROCID'])
    ntasks = int(os.environ['SLURM_NTASKS'])
    node_list = os.environ['SLURM_NODELIST']
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(proc_id % num_gpus)
    addr = subprocess.getoutput('scontrol show hostname {} | head -n1'.format(node_list))
    os.environ['MASTER_PORT'] = str(tcp_port)
    os.environ['MASTER_ADDR'] = addr
    os.environ['WORLD_SIZE'] = str(ntasks)
    os.environ['RANK'] = str(proc_id)
    dist.init_process_group(backend=backend)

    total_gpus = dist.get_world_size()
    rank = dist.get_rank()
    return total_gpus, rank


def init_dist_pytorch(tcp_port, local_rank, backend='nccl'):
    if mp.get_start_method(allow_none=True) is None:
        mp.set_start_method('spawn')

    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(local_rank % num_gpus)
    dist.init_process_group(
        backend=backend,
        init_method='tcp://127.0.0.1:%d' % tcp_port,
        rank=local_rank,
        world_size=num_gpus
    )
    rank = dist.get_rank()
    return num_gpus, rank


def get_dist_info():
    if torch.__version__ < '1.0':
        initialized = dist._initialized
    else:
        if dist.is_available():
            initialized = dist.is_initialized()
        else:
            initialized = False
    if initialized:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1
    return rank, world_size


def merge_results_dist(result_part, size, tmpdir):
    rank, world_size = get_dist_info()
    os.makedirs(tmpdir, exist_ok=True)

    dist.barrier()
    pickle.dump(result_part, open(os.path.join(tmpdir, 'result_part_{}.pkl'.format(rank)), 'wb'))
    dist.barrier()

    if rank != 0:
        return None

    part_list = []
    for i in range(world_size):
        part_file = os.path.join(tmpdir, 'result_part_{}.pkl'.format(i))
        part_list.append(pickle.load(open(part_file, 'rb')))

    ordered_results = []
    for res in zip(*part_list):
        ordered_results.extend(list(res))
    ordered_results = ordered_results[:size]
    shutil.rmtree(tmpdir)
    return ordered_results


def mesh_formulation_torch(points, max_bound, min_bound, grid_size, mode='UD'):
    """
        formulate voxels mesh in torch
        support mode: UD, SID, LID, if using SID or LID, only apply on first dim
        :param points: (N, d), d >= 3, lidar points or gt box in cartesian coords (x, y, z) or polar coords (rho, phi, z)
        :param max_bound: (3,)
        :param min_bound:
        :param grid_size:
        :param mode:
        :return:
        """
    if not isinstance(points, torch.Tensor):
        points = torch.from_numpy(np.asarray(points))
    if not isinstance(max_bound, torch.Tensor):
        max_bound = torch.from_numpy(np.asarray(max_bound))
        min_bound = torch.from_numpy(np.asarray(min_bound))
        grid_size = torch.from_numpy(np.asarray(grid_size))
    crop_range = max_bound - min_bound
    intervals = crop_range / grid_size
    assert (intervals != 0).all()
    points = torch.max(torch.min(points[:, :3], max_bound - 1e-6), min_bound)
    grid_ind = (points - min_bound) / intervals
    grid_ind_int = grid_ind.int()
    voxel_centers = (grid_ind_int.float() + 0.5) * intervals + min_bound
    intervals = intervals.reshape(1, -1).repeat(points.shape[0], 1)
    if mode == 'UD':
        return grid_ind, grid_ind_int, voxel_centers, intervals
    elif mode in ['SID', 'LID']:
        points_rho = points[:, 0]
        if mode == 'SID':
            ind_rho = grid_size[0] * torch.log((1 + points_rho) / (1 + min_bound[0])) / torch.log(
                (1 + max_bound[0]) / (1 + min_bound[0]))
            ind_rho_int = ind_rho.int()

            lower_bound_rho = (1 + min_bound[0]) * torch.exp(
                (ind_rho_int / grid_size[0]) * torch.log((1 + max_bound[0]) / (1 + min_bound[0]))) - 1
            uper_bound_rho = (1 + min_bound[0]) * torch.exp(
                ((ind_rho_int + 1) / grid_size[0]) * torch.log((1 + max_bound[0]) / (1 + min_bound[0]))) - 1
        elif mode == 'LID':
            bin_size = 2 * (max_bound[0] - min_bound[0]) / (grid_size[0] * (1 + grid_size[0]))
            ind_rho = - 0.5 + 0.5 * torch.sqrt(1 + 8 * (points_rho - min_bound[0]) / bin_size)
            ind_rho_int = ind_rho.int()

            lower_bound_rho = min_bound[0] + bin_size * ((2 * ind_rho_int + 1) ** 2 - 1) / 8
            uper_bound_rho = min_bound[0] + bin_size * ((2 * (ind_rho_int + 1) + 1) ** 2 - 1) / 8
        intervals_rho = uper_bound_rho - lower_bound_rho
        voxel_centers_rho = lower_bound_rho + intervals_rho * 0.5

        grid_ind = torch.cat((ind_rho.reshape(-1, 1), grid_ind[:, 1:3]), dim=-1)
        grid_ind_int = torch.cat((ind_rho_int.reshape(-1, 1), grid_ind_int[:, 1:3]), dim=-1)
        voxel_centers = torch.cat((voxel_centers_rho.reshape(-1, 1), voxel_centers[:, 1:3]), dim=-1)
        intervals = torch.cat((intervals_rho.reshape(-1, 1), intervals[:, 1:3]), dim=-1)
        return grid_ind, grid_ind_int, voxel_centers, intervals
    else:
        raise NotImplementedError


def mesh_formulation_numpy(points, max_bound, min_bound, grid_size, mode='UD'):
    """
    formulate voxels mesh in numpy
    support mode: UD, SID, LID, if using SID or LID, only apply on first dim
    :param points: (N, 3), in cartesian or polar coords
    :param max_bound: (3,)
    :param min_bound:
    :param grid_size:
    :param mode:
    :return:
    """
    if not isinstance(points, np.ndarray):
        points = np.asarray(points)
    if not isinstance(max_bound, np.ndarray):
        max_bound = np.asarray(max_bound)
        min_bound = np.asarray(min_bound)
        grid_size = np.asarray(grid_size)
    crop_range = max_bound - min_bound
    intervals = crop_range / grid_size
    assert (intervals != 0).all()
    points = np.clip(points[:, :3], min_bound, max_bound - 1e-6)
    grid_ind = (points - min_bound) / intervals
    grid_ind_int = grid_ind.astype(np.int)
    voxel_centers = (grid_ind_int.astype(np.float32) + 0.5) * intervals + min_bound
    intervals = intervals.reshape(1, -1).repeat(points.shape[0], axis=0)
    if mode == 'UD':
        return grid_ind, grid_ind_int, voxel_centers, intervals
    elif mode in ['SID', 'LID']:
        points_rho = points[:, 0]
        if mode == 'SID':
            ind_rho = grid_size[0] * np.log((1 + points_rho) / (1 + min_bound[0])) / np.log(
                (1 + max_bound[0]) / (1 + min_bound[0]))
            ind_rho_int = ind_rho.astype(np.int)

            lower_bound_rho = (1 + min_bound[0]) * np.exp(
                (ind_rho_int / grid_size[0]) * np.log((1 + max_bound[0]) / (1 + min_bound[0]))) - 1
            uper_bound_rho = (1 + min_bound[0]) * np.exp(
                ((ind_rho_int + 1) / grid_size[0]) * np.log((1 + max_bound[0]) / (1 + min_bound[0]))) - 1
        elif mode == 'LID':
            bin_size = 2 * (max_bound[0] - min_bound[0]) / (grid_size[0] * (1 + grid_size[0]))
            ind_rho = - 0.5 + 0.5 * np.sqrt(1 + 8 * (points_rho - min_bound[0]) / bin_size)
            ind_rho_int = ind_rho.astype(np.int)

            lower_bound_rho = min_bound[0] + bin_size * ((2 * ind_rho_int + 1) ** 2 - 1) / 8
            uper_bound_rho = min_bound[0] + bin_size * ((2 * (ind_rho_int + 1) + 1) ** 2 - 1) / 8
        intervals_rho = uper_bound_rho - lower_bound_rho
        voxel_centers_rho = lower_bound_rho + intervals_rho * 0.5

        grid_ind = np.concatenate((ind_rho.reshape(-1, 1), grid_ind[:, 1:3]), axis=-1)
        grid_ind_int = np.concatenate((ind_rho_int.reshape(-1, 1), grid_ind[:, 1:3]), axis=-1)
        voxel_centers = np.concatenate((voxel_centers_rho.reshape(-1, 1), voxel_centers[:, 1:3]), axis=-1)
        intervals = np.concatenate((intervals_rho.reshape(-1, 1), intervals[:, 1:3]), axis=-1)
        return grid_ind, grid_ind_int, voxel_centers, intervals
    else:
        raise NotImplementedError
