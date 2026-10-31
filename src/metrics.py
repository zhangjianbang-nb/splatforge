"""PSNR and SSIM for image comparison (no external dependency)."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def psnr(a, b):
    mse = F.mse_loss(a, b).clamp_min(1e-10)
    return float(10.0 * torch.log10(1.0 / mse))


def _gaussian_window(size=11, sigma=1.5, device="cpu", dtype=torch.float32):
    coords = torch.arange(size, device=device, dtype=dtype) - (size - 1) / 2.0
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = (g / g.sum())
    return g[:, None] * g[None, :]


def ssim(a, b, size=11, sigma=1.5, data_range=1.0):
    """Structural similarity, matching the standard skimage/scikit formulation."""
    device, dtype = a.device, a.dtype
    w = _gaussian_window(size, sigma, device, dtype).expand(a.shape[1], 1, size, size)
    pad = size // 2
    mu_a = F.conv2d(a, w, padding=pad, groups=a.shape[1])
    mu_b = F.conv2d(b, w, padding=pad, groups=a.shape[1])
    aa, bb, ab = mu_a * mu_a, mu_b * mu_b, mu_a * mu_b
    sig_a = F.conv2d(a * a, w, padding=pad, groups=a.shape[1]) - aa
    sig_b = F.conv2d(b * b, w, padding=pad, groups=a.shape[1]) - bb
    sig_ab = F.conv2d(a * b, w, padding=pad, groups=a.shape[1]) - ab

    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    s = ((2 * ab + c1) * (2 * sig_ab + c2)) / \
        ((aa + bb + c1) * (sig_a + sig_b + c2))
    return float(s.mean().detach())


def dssim_loss(a, b):
    """1 - SSIM, used as the structural term in the training loss."""
    return 1.0 - ssim(a, b)
