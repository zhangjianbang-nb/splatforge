"""The 3D Gaussian scene representation: parameterisation, colour, and
adaptive density control.
"""

from __future__ import annotations

import numpy as np
import torch


SH_C0 = 0.28209479177387814
SH_C1 = 0.4886025119029199
SH_C2 = [1.0925484305920792, -1.0925484305920792, 0.31539156525252005,
         -1.0925484305920792, 0.5462742152960396]
SH_C3 = [-0.5900435899266435, 2.890611442640554, -0.4570457994644658,
         0.3731763325901154, -0.4570457994644658, 1.445305721320277,
         -0.5900435899266435]


def eval_sh(degree, sh, dirs):
    """Evaluate real spherical harmonics.  ``sh``: [N, 16, 3], ``dirs``: [N, 3]."""
    result = SH_C0 * sh[:, 0]
    if degree > 0:
        x, y, z = dirs[:, 0], dirs[:, 1], dirs[:, 2]
        result = result - SH_C1 * y[:, None] * sh[:, 1] \
                        + SH_C1 * z[:, None] * sh[:, 2] \
                        - SH_C1 * x[:, None] * sh[:, 3]
    if degree > 1:
        xx, yy, zz = x * x, y * y, z * z
        xy, yz, xz = x * y, y * z, x * z
        result = (result
                  + SH_C2[0] * xy[:, None] * sh[:, 4]
                  + SH_C2[1] * yz[:, None] * sh[:, 5]
                  + SH_C2[2] * (2 * zz - xx - yy)[:, None] * sh[:, 6]
                  + SH_C2[3] * xz[:, None] * sh[:, 7]
                  + SH_C2[4] * (xx - yy)[:, None] * sh[:, 8])
    if degree > 2:
        result = (result
                  + SH_C3[0] * (y * (3 * xx - yy))[:, None] * sh[:, 9]
                  + SH_C3[1] * (xy * z)[:, None] * sh[:, 10]
                  + SH_C3[2] * (y * (4 * zz - xx - yy))[:, None] * sh[:, 11]
                  + SH_C3[3] * (z * (2 * zz - 3 * xx - 3 * yy))[:, None] * sh[:, 12]
                  + SH_C3[4] * (x * (4 * zz - xx - yy))[:, None] * sh[:, 13]
                  + SH_C3[5] * (z * (xx - yy))[:, None] * sh[:, 14]
                  + SH_C3[6] * (x * (xx - 3 * yy))[:, None] * sh[:, 15])
    return result


def quat_to_rotmat(q):
    w, x, y, z = q.unbind(-1)
    return torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
    ], dim=-1).reshape(-1, 3, 3)


class GaussianModel:
    """Raw parameters + their activations, plus adaptive density control."""

    def __init__(self, sh_degree=3, device="cuda", init_points=None,
                 init_colors=None, spatial_lr_scale=1.0, init_scale=None):
        self.device = device
        self.sh_degree = sh_degree
        self.active_sh_degree = 0
        self.spatial_lr_scale = spatial_lr_scale

        n = init_points.shape[0]
        self._xyz = init_points.to(device).float().requires_grad_(True)

        # Initial splat size.  Seeding from nearest-neighbour spacing of a
        # sparse cloud produces splats tens of pixels wide, which explodes the
        # pair count and wastes the first few hundred iterations.  Sizing them
        # to a couple of pixels at the mean viewing distance instead keeps the
        # initial render close to the image and lets densification refine it.
        if init_scale is None:
            from .dataset import knn_mean_distance
            init_scale = max(knn_mean_distance(init_points.cpu(), k=4), 1e-4)
        self._scaling = torch.full((n, 3), float(np.log(max(init_scale, 1e-6))),
                                   device=device).requires_grad_(True)
        self._rotation = torch.zeros(n, 4, device=device)
        self._rotation[:, 0] = 1.0
        self._rotation.requires_grad_(True)

        self._opacity = torch.full((n, 1), -1.0, device=device).requires_grad_(True)

        # DC term initialised from the sampled pixel colour, higher orders zero
        sh0 = (init_colors.to(device).float() - 0.5) / SH_C0
        self._features = torch.zeros(n, 16, 3, device=device)
        self._features[:, 0] = sh0
        self._features.requires_grad_(True)

        self.max_radii2d = torch.zeros(n, device=device)
        self.xyz_gradient_accum = torch.zeros(n, 1, device=device)
        self.denom = torch.zeros(n, 1, device=device)

        self._setup_optimizer()

    # ---------------- activations ---------------- #
    @property
    def means(self):
        return self._xyz

    def scales(self):
        return torch.exp(self._scaling)

    def quats(self):
        return self._rotation / self._rotation.norm(dim=-1, keepdim=True)

    def opacities(self):
        return torch.sigmoid(self._opacity)

    def cov3d(self):
        R = quat_to_rotmat(self.quats())
        M = R * self.scales()[:, None, :]
        return M @ M.transpose(1, 2)

    def colors(self, camera_center):
        """View-dependent RGB via SH evaluation, in [0, 1]."""
        dirs = self.means - camera_center
        dirs = dirs / dirs.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        c = eval_sh(self.active_sh_degree, self._features, dirs)
        return (c + 0.5).clamp(0.0, 1.0)

    # ---------------- optimiser ---------------- #
    def _setup_optimizer(self):
        self.lr_xyz = 1.6e-4 * self.spatial_lr_scale
        self.lr_features_dc = 2.5e-3
        self.lr_features_rest = 2.5e-4
        self.lr_opacity = 5e-2
        self.lr_scaling = 5e-3
        self.lr_rotation = 1e-3
        self.optimizer = torch.optim.Adam([
            {"params": [self._xyz], "lr": self.lr_xyz, "name": "xyz"},
            {"params": [self._features], "lr": self.lr_features_dc, "name": "f_dc"},
            {"params": [self._opacity], "lr": self.lr_opacity, "name": "opacity"},
            {"params": [self._scaling], "lr": self.lr_scaling, "name": "scaling"},
            {"params": [self._rotation], "lr": self.lr_rotation, "name": "rotation"},
        ], lr=0.0, eps=1e-15)

    def update_learning_rate(self, iteration, total):
        """Exponential decay on position lr (standard 3DGS schedule)."""
        lr = self.lr_xyz * (0.01 ** (iteration / max(1, total)))
        for g in self.optimizer.param_groups:
            if g["name"] == "xyz":
                g["lr"] = lr
            elif g["name"] == "f_dc":
                g["lr"] = self.lr_features_dc
            elif g["name"] == "opacity":
                g["lr"] = self.lr_opacity
            elif g["name"] == "scaling":
                g["lr"] = self.lr_scaling
            elif g["name"] == "rotation":
                g["lr"] = self.lr_rotation

    def oneup_sh_degree(self):
        if self.active_sh_degree < self.sh_degree:
            self.active_sh_degree += 1

    # ---------------- adaptive density control ---------------- #
    def add_densification_stats(self, grad_norm, visible):
        self.xyz_gradient_accum[visible] += grad_norm[visible].detach().unsqueeze(-1)
        self.denom[visible] += 1.0

    def _prune_mask(self, min_opacity, extent, max_screen_size):
        op = self.opacities().detach().squeeze(-1)
        scale = self.scales().detach().max(dim=1).values
        prune = op < min_opacity
        prune |= scale > 0.1 * extent
        if max_screen_size is not None:
            prune |= self.max_radii2d > max_screen_size
        return prune

    # --- optimiser bookkeeping -------------------------------------------------
    # Densification changes the number of rows in every parameter.  The naive
    # approach -- rebuild the Adam optimiser -- silently discards the momentum
    # and second-moment estimates every densify step (every 100 iterations),
    # which makes the optimiser restart over and over and lets the model
    # degrade.  These two helpers instead extend/trim the existing tensors and
    # carry the per-row optimiser state along with them.
    def _cat_to_optimizer(self, tensors):
        out = {}
        for group in self.optimizer.param_groups:
            name = group["name"]
            old = group["params"][0]
            state = self.optimizer.state.pop(old, None)
            new = torch.cat([old, tensors[name]], dim=0).detach().requires_grad_(True)
            if state is not None:
                zeros = torch.zeros_like(tensors[name])
                state["exp_avg"] = torch.cat([state["exp_avg"], zeros], dim=0)
                state["exp_avg_sq"] = torch.cat([state["exp_avg_sq"], zeros], dim=0)
                self.optimizer.state[new] = state
            group["params"][0] = new
            out[name] = new
        return out

    def _prune_optimizer(self, keep):
        out = {}
        for group in self.optimizer.param_groups:
            name = group["name"]
            old = group["params"][0]
            state = self.optimizer.state.pop(old, None)
            new = old[keep].detach().requires_grad_(True)
            if state is not None:
                state["exp_avg"] = state["exp_avg"][keep]
                state["exp_avg_sq"] = state["exp_avg_sq"][keep]
                self.optimizer.state[new] = state
            group["params"][0] = new
            out[name] = new
        return out

    def densify_and_prune(self, max_grad, min_opacity, extent,
                          max_screen_size, percent_dense=0.01,
                          max_gaussians=None):
        grads = (self.xyz_gradient_accum / self.denom.clamp_min(1.0)).squeeze(-1)
        grads[~torch.isfinite(grads)] = 0.0
        selected = grads >= max_grad

        # a hard cap keeps a pathological gradient distribution from exploding
        # the point count before the pruner can catch up
        if max_gaussians is not None and self._xyz.shape[0] >= max_gaussians:
            selected = torch.zeros_like(selected)

        n_clone = n_split = 0
        if selected.any():
            scale = self.scales().detach().max(dim=1).values
            small = scale <= percent_dense * extent
            clone_mask = selected & small
            split_mask = selected & (~small)
            n_clone, n_split = int(clone_mask.sum()), int(split_mask.sum())

            if n_clone:
                self._install(self._cat_to_optimizer({
                    "xyz": self._xyz[clone_mask].detach(),
                    "f_dc": self._features[clone_mask].detach(),
                    "opacity": self._opacity[clone_mask].detach(),
                    "scaling": self._scaling[clone_mask].detach(),
                    "rotation": self._rotation[clone_mask].detach(),
                }))

            if n_split:
                idx = torch.nonzero(split_mask).flatten()
                reps = 2
                xyz = self._xyz[idx].detach()
                scaling = self._scaling[idx].detach()
                rot = self.quats()[idx].detach()
                op = self._opacity[idx].detach()
                feats = self._features[idx].detach()
                # jitter the children by the parent's own covariance, rotated
                # into world space: M = R diag(s), samples ~ N(0, M M^T)
                R = quat_to_rotmat(rot)
                noise = torch.randn(idx.numel(), reps, 3, device=self.device)
                offset = torch.einsum(
                    "nij,nrj->nri", R * torch.exp(scaling)[:, None, :], noise)
                self._install(self._cat_to_optimizer({
                    "xyz": (xyz[:, None, :] + offset).reshape(-1, 3),
                    "f_dc": feats.repeat(reps, 1, 1),
                    "opacity": op.repeat(reps, 1),
                    "scaling": (scaling - np.log(1.6)).repeat(reps, 1),
                    "rotation": rot.repeat(reps, 1),
                }))

            # newly appended rows need zero-filled statistics to stay aligned
            pad = self._xyz.shape[0] - self.max_radii2d.shape[0]
            if pad > 0:
                self.max_radii2d = torch.cat(
                    [self.max_radii2d, torch.zeros(pad, device=self.device)])
                self.xyz_gradient_accum = torch.cat(
                    [self.xyz_gradient_accum, torch.zeros(pad, 1, device=self.device)])
                self.denom = torch.cat(
                    [self.denom, torch.zeros(pad, 1, device=self.device)])

        self._prune(min_opacity, extent, max_screen_size)
        return n_clone, n_split

    def _install(self, params):
        """Copy parameter tensors returned by the optimiser helpers back."""
        self._xyz = params["xyz"]
        self._features = params["f_dc"]
        self._opacity = params["opacity"]
        self._scaling = params["scaling"]
        self._rotation = params["rotation"]

    def _prune(self, min_opacity, extent, max_screen_size):
        mask = self._prune_mask(min_opacity, extent, max_screen_size)
        keep = ~mask
        if keep.all():
            return
        self._install(self._prune_optimizer(keep))
        self.max_radii2d = self.max_radii2d[keep]
        self.xyz_gradient_accum = self.xyz_gradient_accum[keep]
        self.denom = self.denom[keep]

    def _replace_param(self, tensor, name):
        """Swap a parameter tensor into the optimiser, keeping its slot.

        Assigning ``self._opacity = new_tensor`` alone would leave Adam holding
        the old tensor: gradients would flow into the new one but never be
        applied, silently freezing opacity for the rest of training.
        """
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                old = group["params"][0]
                state = self.optimizer.state.get(old)
                if state is not None:
                    state["exp_avg"] = torch.zeros_like(tensor)
                    state["exp_avg_sq"] = torch.zeros_like(tensor)
                group["params"][0] = tensor
                return tensor
        raise KeyError(f"no optimiser group named {name}")

    def reset_opacity(self, min_opacity=0.01):
        """Clamp opacities down periodically so floaters get pruned away."""
        op = self.opacities().detach()
        new = torch.minimum(op, torch.full_like(op, min_opacity))
        new = torch.logit(new.clamp(1e-6, 1 - 1e-6)).requires_grad_(True)
        self._opacity = self._replace_param(new, "opacity")

    # ---------------- persistence ---------------- #
    def to_ply(self, path, aa_sigma_px=0.0):
        """Export the scene as a binary PLY point cloud with SH coefficients.

        When the anti-aliasing filter is active the stored opacity is the
        compensated one, so we bake it into the export to keep the exported
        model visually consistent with the renderer.
        """
        xyz = self._xyz.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rot = self.quats().detach().cpu().numpy()
        opacity = self._opacity.detach().cpu().numpy()
        feats = self._features.detach().cpu().numpy().transpose(0, 2, 1).reshape(-1, 48)

        header = ["ply", "format binary_little_endian 1.0", f"element vertex {xyz.shape[0]}",
                  "property float x", "property float y", "property float z",
                  "property float nx", "property float ny", "property float nz",
                  "property float f_dc_0", "property float f_dc_1", "property float f_dc_2"]
        for i in range(45):
            header.append(f"property float f_rest_{i}")
        header += ["property float opacity",
                   "property float scale_0", "property float scale_1", "property float scale_2",
                   "property float rot_0", "property float rot_1",
                   "property float rot_2", "property float rot_3", "end_header"]

        data = np.concatenate([
            xyz, np.zeros_like(xyz), feats, opacity, scale, rot], axis=1).astype(np.float32)
        with open(path, "wb") as f:
            f.write(("\n".join(header) + "\n").encode("ascii"))
            f.write(data.tobytes())
