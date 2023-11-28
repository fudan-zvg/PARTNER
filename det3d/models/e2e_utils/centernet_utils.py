import torch
import numpy as np


def gaussian_radius(height, width, min_overlap=0.5):
    """
    Args:
        height: (N)
        width: (N)
        min_overlap:
    Returns:
    """
    a1 = 1
    b1 = (height + width)
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = (b1 ** 2 - 4 * a1 * c1).sqrt()
    r1 = (b1 + sq1) / 2

    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = (b2 ** 2 - 4 * a2 * c2).sqrt()
    r2 = (b2 + sq2) / 2

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    sq3 = (b3 ** 2 - 4 * a3 * c3).sqrt()
    r3 = (b3 + sq3) / 2
    ret = torch.min(torch.min(r1, r2), r3)
    return ret


def gaussian2D(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def draw_gaussian_to_heatmap(heatmap, center, radius, k=1, valid_mask=None):
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)

    x, y = int(center[0]), int(center[1])

    height, width = heatmap.shape[0:2]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = torch.from_numpy(
        gaussian[radius - top:radius + bottom, radius - left:radius + right]
    ).to(heatmap.device).float()

    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:  # TODO debug
        if valid_mask is not None:
            cur_valid_mask = valid_mask[y - top:y + bottom, x - left:x + right]
            masked_gaussian = masked_gaussian * cur_valid_mask.float()

        torch.max(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap


def draw_center_to_votemap(votemap, center_int, center, drho, dphi, task_class, gaussian_overlap=0.1):
    rho, phi = center_int[0], center_int[1]
    height, width = votemap.shape[:2]

    radius_rho = int(gaussian_radius(drho, drho, min_overlap=gaussian_overlap))
    radius_phi = int(gaussian_radius(dphi, dphi, min_overlap=gaussian_overlap))

    left, right = min(rho, radius_rho), min(width - rho, radius_rho + 1)
    top, bottom = min(phi, radius_phi), min(height - phi, radius_phi + 1)
    votemap[phi - top:phi + bottom, rho - left:rho + right, :4] = center

    diameter_rho = 2 * radius_rho + 1
    diameter_phi = 2 * radius_phi + 1
    diameter = max(diameter_rho, diameter_phi)
    gaussian = gaussian2D((diameter_phi, diameter_rho), sigma=diameter / 6)
    masked_heatmap = votemap[phi - top:phi + bottom, rho - left: rho + right, 4+task_class]
    masked_gaussian = torch.from_numpy(
        gaussian[radius_phi - top:radius_phi + bottom, radius_rho - left:radius_rho + right]).to(masked_heatmap.device)
    torch.max(masked_heatmap, masked_gaussian, out=masked_heatmap)

    return votemap
