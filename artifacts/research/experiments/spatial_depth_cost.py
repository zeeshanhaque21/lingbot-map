"""Experimental mean-field regularization of photometric log-depth costs."""

import torch


def spatial_costs(volume, prior, rgb, factors):
    """Encourage matching absolute depth, with image/depth edge attenuation.

    Labels represent multiplicative corrections to different per-pixel priors.
    Neighbor probabilities must therefore be shifted before comparing depths.
    """
    labels, height, width = volume.shape
    log_depth = prior.clamp_min(1e-8).log()
    log_factors = factors.log()
    step = log_factors[1] - log_factors[0]
    compatibility = torch.exp(
        -0.5 * ((log_factors[:, None] - log_factors[None]) / 0.05).square()
    )
    label_grid = torch.arange(labels, device=volume.device)[:, None, None]
    edges = []
    for dy, dx in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
        neighbor_depth = torch.roll(log_depth, (dy, dx), (0, 1))
        neighbor_rgb = torch.roll(rgb, (dy, dx), (1, 2))
        weight = torch.exp(
            -(rgb - neighbor_rgb).abs().mean(0) / 0.1
            - (log_depth - neighbor_depth).abs() / 0.2
        )
        if dy == 1:
            weight[0] = 0
        elif dy == -1:
            weight[-1] = 0
        elif dx == 1:
            weight[:, 0] = 0
        else:
            weight[:, -1] = 0
        # Shift the neighbor's labels to the current pixel's absolute depth.
        coordinate = label_grid + (log_depth - neighbor_depth)[None] / step
        lower = coordinate.floor()
        fraction = coordinate - lower
        valid = (coordinate >= 0) & (coordinate <= labels - 1)
        edges.append(
            (
                dy,
                dx,
                weight,
                lower.long().clamp(0, labels - 1),
                (lower.long() + 1).clamp(0, labels - 1),
                fraction,
                valid,
            )
        )
    unary = volume.clamp(max=1)
    posterior = torch.softmax(-unary / 0.015, dim=0)
    for _ in range(4):
        compatible = (compatibility @ posterior.reshape(labels, -1)).reshape(
            labels, height, width
        )
        message, denominator = torch.zeros_like(volume), torch.zeros_like(volume)
        for dy, dx, weight, lower, upper, fraction, valid in edges:
            neighbor = torch.roll(compatible, (dy, dx), (1, 2))
            aligned = (1 - fraction) * torch.gather(
                neighbor, 0, lower
            ) + fraction * torch.gather(neighbor, 0, upper)
            effective = weight[None] * valid
            message += aligned * effective
            denominator += effective
        # Average strong neighbors without amplifying weak links across edges.
        energy = unary - 0.08 * message / denominator.clamp_min(1)
        posterior = torch.softmax(-energy / 0.015, dim=0)
    return energy
