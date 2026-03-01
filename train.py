import json
from PIL import Image
import glob
import argparse
import torch
import open3d as o3d
import numpy as np
from tqdm import tqdm
from model import SDFNet
from loss import compute_loss
import matplotlib.pyplot as plt
import torchvision.transforms as T
from copy import deepcopy
from utils import str2bool

from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import autocast, GradScaler
from lion_pytorch import Lion

# ========== 這幾行是為了 K-FAC Minimal ==========
from typing import Optional, Iterable, Dict, List, Tuple
import torch.nn as nn
from dataclasses import dataclass
# ==============================================

# set the random seed for reproducibility
import torch, numpy as np, random, os
torch.manual_seed(10)
np.random.seed(10)
random.seed(10)
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")

# ===========================
# K-FAC (Linear-only) Minimal
# ===========================
@dataclass
class _KFACConfig:
    lr: float = 5e-3
    ema_decay: float = 0.95
    damping: float = 1e-3
    weight_decay: float = 0.0
    stat_update_freq: int = 1
    inv_update_freq: int = 10
    kl_clip: Optional[float] = None

class _KFACLinearHandles:
    def __init__(self, module: nn.Linear, name: str):
        self.module = module
        self.name = name
        self.a = None   # E[x x^T] (FP32)
        self.g = None   # E[dy dy^T] (FP32)
        self.a_inv = None
        self.g_inv = None
        self.hook_in = None
        self.hook_out = None

    def register(self, ema_decay: float):
        def _fwd_hook(mod, inp, outp):
            x = inp[0].detach()
            x = x.reshape(-1, x.shape[-1]).to(torch.float32)
            aa = (x.t() @ x) / max(1, x.shape[0])
            if self.a is None:
                self.a = aa
            else:
                self.a.mul_(ema_decay).add_(aa, alpha=1 - ema_decay)

        def _bwd_hook(mod, grad_input, grad_output):
            gy = grad_output[0].detach()
            gy = gy.reshape(-1, gy.shape[-1]).to(torch.float32)
            gg = (gy.t() @ gy) / max(1, gy.shape[0])
            if self.g is None:
                self.g = gg
            else:
                self.g.mul_(ema_decay).add_(gg, alpha=1 - ema_decay)

        self.hook_in  = self.module.register_forward_hook(_fwd_hook)
        self.hook_out = self.module.register_full_backward_hook(_bwd_hook)

    def remove(self):
        if self.hook_in is not None:
            self.hook_in.remove()
        if self.hook_out is not None:
            self.hook_out.remove()

class KFACLinearOnly(torch.optim.Optimizer):
    """K-FAC for nn.Linear layers under module names containing given filters (e.g., 'geo_heads', 'color_heads')."""
    def __init__(self, model: nn.Module, module_name_filters: Iterable[str],
                 lr: float = 5e-3, ema_decay: float = 0.95, damping: float = 1e-3,
                 weight_decay: float = 0.0, stat_update_freq: int = 1, inv_update_freq: int = 10,
                 kl_clip: Optional[float] = None):
        self.model = model
        self.cfg = _KFACConfig(lr, ema_decay, damping, weight_decay, stat_update_freq, inv_update_freq, kl_clip)
        params = [p for n, p in model.named_parameters() if any(f in n for f in module_name_filters)]
        super().__init__([{'params': params}], dict(lr=lr))
        self.modules: Dict[str, _KFACLinearHandles] = {}
        for name, m in model.named_modules():
            if any(f in name for f in module_name_filters) and isinstance(m, nn.Linear):
                h = _KFACLinearHandles(m, name)
                h.register(ema_decay)
                self.modules[name] = h
        self._step = 0

    @torch.no_grad()
    def step(self, closure=None):
        self._step += 1

        if self.cfg.weight_decay and self.cfg.weight_decay > 0:
            for g in self.param_groups:
                for p in g['params']:
                    if p.grad is None:
                        continue
                    p.add_(p, alpha=-self.cfg.weight_decay * self.param_groups[0]['lr'])

        need_inv = (self._step % self.cfg.inv_update_freq == 0)

        with autocast(enabled=False):
            for name, h in self.modules.items():
                m = h.module
                if m.weight.grad is None:
                    continue

                if need_inv and (h.a is not None) and (h.g is not None):
                    def _spd_inverse(M: torch.Tensor, base_damp: float) -> torch.Tensor:
                        M = M.to(torch.float32)
                        M = 0.5 * (M + M.t())
                        eye = torch.eye(M.shape[0], device=M.device, dtype=M.dtype)
                        scale = torch.clamp(M.diag().mean(), min=1e-6)
                        Mn = M / scale
                        jitter = base_damp / scale
                        for _ in range(6):
                            try:
                                L = torch.linalg.cholesky(Mn + jitter * eye)
                                inv = torch.cholesky_inverse(L)
                                inv = inv / scale
                                return 0.5 * (inv + inv.t())
                            except Exception:
                                jitter *= 10.0
                        try:
                            evals, evecs = torch.linalg.eigh(Mn)
                            evals_clamped = torch.clamp(evals, min=max(jitter, 1e-6))
                            inv = (evecs @ torch.diag(1.0 / evals_clamped) @ evecs.t())
                            inv = inv / scale
                            return 0.5 * (inv + inv.t())
                        except Exception:
                            pass
                        try:
                            U, S, Vh = torch.linalg.svd(Mn, full_matrices=False)
                            S_clamped = torch.clamp(S, min=max(jitter, 1e-6))
                            inv = (Vh.transpose(-2, -1) @ torch.diag(1.0 / S_clamped) @ U.transpose(-2, -1))
                            inv = inv / scale
                            return 0.5 * (inv + inv.t())
                        except Exception:
                            return (1.0 / max(base_damp, 1e-6)) * eye

                    h.a_inv = _spd_inverse(h.a, self.cfg.damping)
                    h.g_inv = _spd_inverse(h.g, self.cfg.damping)

                if (h.a_inv is not None) and (h.g_inv is not None):
                    gw = m.weight.grad
                    pre_gw = h.g_inv @ gw.to(torch.float32) @ h.a_inv
                    pre_gw = torch.nan_to_num(pre_gw, nan=0.0, posinf=1e6, neginf=-1e6)
                    m.weight.grad.copy_(pre_gw.to(gw.dtype))

                    if m.bias is not None and m.bias.grad is not None:
                        gb = m.bias.grad.view(-1, 1).to(torch.float32)
                        pre_gb = (h.g_inv @ gb).view_as(m.bias.grad)
                        pre_gb = torch.nan_to_num(pre_gb, nan=0.0, posinf=1e6, neginf=-1e6)
                        m.bias.grad.copy_(pre_gb.to(m.bias.grad.dtype))

        max_norm = 1.0
        all_grads = [p.grad.detach().norm() for g in self.param_groups for p in g['params'] if (p.grad is not None)]
        if len(all_grads) > 0:
            total_norm = torch.norm(torch.stack(all_grads), p=2)
            if not torch.isfinite(total_norm):
                return None
            if total_norm > max_norm and total_norm > 0:
                scale = max_norm / (total_norm + 1e-12)
                for g in self.param_groups:
                    for p in g['params']:
                        if p.grad is not None:
                            p.grad.mul_(scale)

        lr = self.param_groups[0]['lr']
        for g in self.param_groups:
            for p in g['params']:
                if p.grad is None:
                    continue
                p.add_(p.grad, alpha=-lr)
        return None

    def close(self):
        for h in self.modules.values():
            h.remove()

def load_schedule(schedule_path):
    with open(schedule_path, 'r') as f:
        return json.load(f)

parser = argparse.ArgumentParser(description="SDFNet training script.")
parser.add_argument('--lr', type=float, default=0.005, help="Learning rate.")
parser.add_argument('--desc', type=str, required=True, help="Experiment description.")
parser.add_argument('--log_path', type=str, required=True, help="Log file path.")
parser.add_argument('--ckpt_path', type=str, required=True, help="Checkpoint save path.")
parser.add_argument('--preprocess', type=str, required=True, help="Preprocessed method.")
parser.add_argument('--file_name', type=str, required=True, help="Pointcloud file name.")
parser.add_argument('--schedule_path', type=str, required=True, help="Training schedule file name.")
parser.add_argument('--is_a100', type=str2bool, required=True, help="Training on A100 or not.")

# === K-FAC 相關超參數 ===
parser.add_argument('--kfac_head_lr_mul', type=float, default=5.0, help='K-FAC 啟動後 heads 的 LR 倍數（相對於 --lr）')
parser.add_argument('--kfac_damping', type=float, default=8e-3, help='K-FAC damping')
parser.add_argument('--kfac_inv_update', type=int, default=20, help='K-FAC inverse 更新頻率')
parser.add_argument('--kfac_ema', type=float, default=0.95, help='K-FAC 統計的 EMA 係數')

args = parser.parse_args()

lr = args.lr
is_a100 = args.is_a100
file_name = args.file_name
log_path = args.log_path
ckpt_path = args.ckpt_path
sche_path = args.schedule_path
preprocess = args.preprocess

kfac_head_lr_mul = args.kfac_head_lr_mul
kfac_damping = args.kfac_damping
kfac_inv_update = args.kfac_inv_update
kfac_ema = args.kfac_ema

# ---------- Device ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Using device: {device}")

# Load training schedule
schedule = load_schedule(sche_path)
total_epochs = schedule["total_epochs"]
train_cfg = schedule["train"]
kfac_start_epoch = 0.6 * total_epochs

# Read pcd
pointcloud_path = f"data/{preprocess}_preprocessed/{file_name}/output_pointcloud_normal.ply"
pcd = o3d.io.read_point_cloud(pointcloud_path)

# to torch
points_np   = np.asarray(pcd.points, dtype=np.float32)   # (N,3)
gt_colors_np = np.asarray(pcd.colors, dtype=np.float32)  # (N,3)
normals   = np.asarray(pcd.normals, dtype=np.float32) # (N,3)

points   = torch.tensor(points_np,   dtype=torch.float32, device=device)  # (N,3)
gt_colors = torch.tensor(gt_colors_np, dtype=torch.float32, device=device)

# === normalize to [0,1]^3 with uniform scale ===
mins   = points.amin(dim=0, keepdim=True)     # (1,3)
maxs   = points.amax(dim=0, keepdim=True)     # (1,3)
centre = (maxs + mins) * 0.5                  # (1,3)
scale  = (maxs - mins).max() + 1e-8           # scalar (最大邊長，避免除以0)

points = (points - centre) / scale + 0.5      # → roughly [0,1]^3（等比縮放 + 平移）

# (可選) 若你想完全落在 [0,1] 而不是“約略”，可再 clamp：
# points = points.clamp_(0.0, 1.0)

x = points.to(device)   # [N,3] 直接當輸入
model = SDFNet().to(device)

# ========= Optimizers / Schedulers =========
# 先用 Lion 訓練全部參數；建立「唯一」的 Cosine 調度器
optimizer_all = Lion(model.parameters(), lr=lr, weight_decay=1e-2)
scheduler_cosine = CosineAnnealingLR(optimizer_all, T_max=total_epochs, eta_min=lr / 1000.0)

# 之後切換用的（初始為 None）
use_kfac = False
optimizer_encoder = None
optimizer_heads = None  # K-FAC
# 不再使用多個 scheduler，僅此一個：
# scheduler_cosine 會在切換時動態改綁 optimizer

scaler = GradScaler()

# training
model.train()
pbar = tqdm(range(total_epochs), desc="Training")

with open(log_path, "w") as f:
    f.write("epoch,loss_total,loss_sdf,loss_zero,loss_eikonal,loss_normal,loss_sparse,loss_color_geo,loss_neg_sdf,loss_singular_hessian,loss_rgb_gt,learning_rate\n")

for epoch in pbar:
    # 若到達切換點，建立混合優化器（保持同一個 cosine 週期）
    if (not use_kfac) and (epoch >= kfac_start_epoch):
        print(f"[Switch] Enable K-FAC at epoch {epoch}")
        use_kfac = True

        # 目前 cosine 的 LR（保持週期連續）
        cur_lr = scheduler_cosine.get_last_lr()[0]

        # Encoder → Lion（沿用當前 LR）
        encoder_params = [p for n, p in model.named_parameters()
                          if ('geo_head' not in n and 'color_head' not in n)]
        optimizer_encoder = Lion(encoder_params, lr=cur_lr, weight_decay=1e-2)

        # Heads → K-FAC（LR = 倍數 × 當前 LR）
        head_lr = kfac_head_lr_mul * cur_lr
        optimizer_heads = KFACLinearOnly(
            model=model,
            module_name_filters=['geo_head', 'color_head'],
            lr=head_lr,
            ema_decay=kfac_ema,
            damping=kfac_damping,
            stat_update_freq=1,
            inv_update_freq=kfac_inv_update
        )

        # **關鍵**：不要新建 scheduler，直接把同一個 cosine 綁到新的 encoder optimizer
        scheduler_cosine.optimizer = optimizer_encoder

        # 不再用 warmup 的 optimizer
        del optimizer_all  # scheduler_cosine 已經改綁，不再需要它

    sigma = train_cfg["sigma"]

    weight_sdf                        = train_cfg["loss_weights"]["loss_sdf"]
    weight_zero                       = train_cfg["loss_weights"]["loss_zero"]
    eik_init                          = train_cfg["loss_weights"]["loss_eikonal"]["loss_eikonal_init"]
    weight_eikonal_final              = train_cfg["loss_weights"]["loss_eikonal"]["loss_eikonal_final"]
    eik_ramp                          = train_cfg["loss_weights"]["loss_eikonal"]["loss_eikonal_ramp"]
    weight_glob_eik                   = train_cfg["loss_weights"]["loss_eikonal"]["loss_global_eikonal"]
    weight_normal                     = train_cfg["loss_weights"]["loss_normal"]
    weight_sparse                     = train_cfg["loss_weights"]["loss_sparse"]
    weight_color_geo                  = train_cfg["loss_weights"]["loss_color_geo"]
    weight_neg_sdf                    = train_cfg["loss_weights"]["loss_neg_sdf"]
    singular_hessian_init             = train_cfg["loss_weights"]["loss_singular_hessian"]["loss_singular_hessian_init"]
    weight_singular_hessian_final     = train_cfg["loss_weights"]["loss_singular_hessian"]["loss_singular_hessian_final"]
    singular_hessian_ramp             = train_cfg["loss_weights"]["loss_singular_hessian"]["loss_singular_hessian_ramp"]
    weight_rgb_gt                     = train_cfg["loss_weights"]["loss_rgb_gt"]

    # 產生 xyz 的 noise
    epsilon = torch.randn_like(x[:, :3]) * sigma
    x_noisy_full = torch.cat([x[:, :3] + epsilon], dim=1)  # [N, 3]
    epsilon = torch.cat([epsilon], dim=1)                  # [N, 3]
    x_noisy_full = x_noisy_full.to(device)
    epsilon = epsilon.to(device)

    # 清梯度
    if use_kfac:
        optimizer_encoder.zero_grad(set_to_none=True)
        optimizer_heads.zero_grad(set_to_none=True)
    else:
        # 還在 warmup：用整體的 Lion
        scheduler_cosine.optimizer.zero_grad(set_to_none=True)

    with autocast():
        (loss_sdf, loss_zero, loss_eikonal_surface, loss_eikonal_global, 
         loss_normal, loss_sparse, loss_color_geo, loss_neg_sdf, loss_singular_hessian, 
         loss_rgb_gt
        ) = compute_loss(
            model,
            x,
            gt_colors,
            x_noisy_full,
            epsilon,
            normals,
            epoch,
            total_epochs,
            is_a100,
            weight_sdf,
            weight_zero,
            eik_init,
            weight_glob_eik,
            weight_normal,
            weight_sparse,
            weight_color_geo,
            weight_neg_sdf,
            singular_hessian_init,
            weight_rgb_gt
        )

        w_eik = eik_init + (weight_eikonal_final - eik_init) * min(epoch / eik_ramp, 1.0)
        weight_singular_hessian = singular_hessian_init + (weight_singular_hessian_final - singular_hessian_init) * min(epoch / singular_hessian_ramp, 1.0)
        loss_total = weight_sdf * loss_sdf \
                    + weight_zero * loss_zero \
                    + w_eik * loss_eikonal_surface \
                    + weight_glob_eik * loss_eikonal_global \
                    + weight_normal * loss_normal \
                    + weight_sparse * loss_sparse \
                    + weight_color_geo * loss_color_geo \
                    + weight_neg_sdf * loss_neg_sdf \
                    + weight_singular_hessian * loss_singular_hessian \
                    + weight_rgb_gt * loss_rgb_gt

    loss_eikonal = w_eik * loss_eikonal_surface + weight_glob_eik * loss_eikonal_global

    scaler.scale(loss_total).backward()

    # === 更新 ===
    if use_kfac:
        # 讓 K-FAC/Lion 看未縮放梯度
        scaler.unscale_(optimizer_encoder)
        scaler.unscale_(optimizer_heads)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # K-FAC 先更新 heads
        optimizer_heads.step()

        # Lion 更新 encoder
        scaler.step(optimizer_encoder)
        scaler.update()

        current_lr = scheduler_cosine.get_last_lr()[0]
    else:
        # warmup：整體 Lion（scheduler 綁在 optimizer_all 上）
        scaler.unscale_(scheduler_cosine.optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(scheduler_cosine.optimizer)
        scaler.update()
        current_lr = scheduler_cosine.get_last_lr()[0]

    # **唯一** 的 cosine 調度器步進（整個訓練期間只呼叫它）
    scheduler_cosine.step()

    pbar.set_postfix(loss=loss_total.item(), lr=current_lr)

    # log each component
    with open(log_path, "a") as f:
        f.write(f"{epoch}, \
                {loss_total.item():.6f}, \
                {loss_sdf.item():.6f}, \
                {loss_zero.item():.6f}, \
                {loss_eikonal.item():.6f}, \
                {loss_normal.item():.6f}, \
                {loss_sparse.item():.6f}, \
                {loss_color_geo.item():.6f}, \
                {loss_neg_sdf.item():.6f}, \
                {loss_singular_hessian.item():.6f}, \
                {loss_rgb_gt.item():.6f}, \
                {current_lr:.8f}\n")

    if epoch % 50 == 0 and epoch > 0:
        torch.save({
                "model_state_dict": model.state_dict()
        }, f"{ckpt_path}_epoch{epoch}.pt")

torch.save({
            "model_state_dict": model.state_dict()
        }, f"{ckpt_path}_final.pt")

# 若有啟動 K-FAC，記得移除 hooks
if use_kfac and optimizer_heads is not None:
    optimizer_heads.close()

print("Training finished.")
