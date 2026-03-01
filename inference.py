import argparse
import os
import torch
import numpy as np
from model import SDFNet
from tqdm import tqdm
from scipy.ndimage import gaussian_filter
from utils import build_coords_slab_on_device, save_slice_with_axes, save_gradient_norm_slice, marching_cubes_and_export

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fast SDFNet inference with GPU-native coords & slab batching.")
    parser.add_argument('--res', type=int, default=256, help="Grid resolution.")
    parser.add_argument('--ckpt_path', type=str, required=True, help="Model checkpoint path.")
    parser.add_argument('--output_mesh', type=str, required=True, help="Output mesh path prefix.")
    parser.add_argument('--preprocess', type=str, required=True, help="Preprocess method.")
    parser.add_argument('--file_name', type=str, required=True, help="Filename reference.")
    parser.add_argument('--batch_size', type=int, default=2**20, help="Per-forward batch size (points).")
    parser.add_argument('--slab_depth', type=int, default=32, help="Number of z-slices per slab.")
    parser.add_argument('--precision', type=str, default='auto', choices=['fp32','fp16','bf16','auto'],
                        help="AMP precision for inference.")
    parser.add_argument('--sigma', type=float, default=0.5, help="Gaussian smoothing sigma.")
    parser.add_argument('--vtx_batch', type=int, default=2**20, help="Batch size for vertex colorization.")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        try:
            torch.set_float32_matmul_precision('high')  # Ampere+ 可能受益
        except Exception:
            pass

    checkpoint = torch.load(args.ckpt_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model = SDFNet().to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        raise ValueError("Invalid checkpoint format.")

    model.eval()

    res = args.res
    min_bound = [0.0, 0.0, 0.0]
    max_bound = [1.0, 1.0, 1.0]

    # linspace on CPU for spacing（供 marching_cubes 使用）
    x_vals_np = np.linspace(min_bound[0], max_bound[0], res, dtype=np.float32)
    y_vals_np = np.linspace(min_bound[1], max_bound[1], res, dtype=np.float32)
    z_vals_np = np.linspace(min_bound[2], max_bound[2], res, dtype=np.float32)

    # 同步在 GPU 建立 torch 版本（避免 CPU->GPU 巨量拷貝）
    x_vals = torch.linspace(min_bound[0], max_bound[0], res, device=device, dtype=torch.float32)
    y_vals = torch.linspace(min_bound[1], max_bound[1], res, device=device, dtype=torch.float32)
    z_vals = torch.linspace(min_bound[2], max_bound[2], res, device=device, dtype=torch.float32)

    # 預先配置 SDF grid（CPU, float32）
    sdf_grid = np.zeros((res, res, res), dtype=np.float32)

    total_voxels = res * res * res
    slab = max(1, min(args.slab_depth, res))
    bs = max(1, args.batch_size)

    # AMP 設定
    if args.precision == 'fp16' and device.type == 'cuda':
        amp_dtype = torch.float16
    elif args.precision == 'bf16' and device.type == 'cuda':
        amp_dtype = torch.bfloat16
    elif args.precision == 'auto' and device.type == 'cuda':
        # 讓 PyTorch 自行選擇；這裡選擇 bfloat16 較穩定（若卡支援），否則關閉
        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        amp_dtype = None  # 關閉 autocast

    if amp_dtype is not None and device.type == 'cuda':
        autocast_ctx = torch.cuda.amp.autocast(dtype=amp_dtype)
    else:
        autocast_ctx = torch.cuda.amp.autocast(enabled=False)

    # ---- 主推論：以 Z-slab 為單位在 GPU 端生成座標並分批前向 ----
    with torch.inference_mode(), autocast_ctx:
        for z0 in tqdm(range(0, res, slab), desc="Evaluating SDF grid (slabs)"):
            z1 = min(z0 + slab, res)
            # GPU 上建立 slab 座標
            slab_pts = build_coords_slab_on_device(x_vals, y_vals, z_vals, z0, z1, device)  # (Dz*res*res, 3)

            # 分批前向
            slab_sdfs = torch.empty((slab_pts.shape[0],), dtype=torch.float32, device=device)
            for i in range(0, slab_pts.shape[0], bs):
                j = min(i + bs, slab_pts.shape[0])
                sdf_pred, _ = model(slab_pts[i:j])
                # 以 float32 暫存，避免 AMP 取樣誤差累積
                slab_sdfs[i:j] = sdf_pred.reshape(-1).float()

            # write back to CPU numpy (non_blocking)
            slab_sdfs_cpu = slab_sdfs.detach().cpu().numpy()  # non_blocking: need pinned memory; here simplified
            # reshape to (Dz, Ry, Rx)
            slab_sdfs_cpu = slab_sdfs_cpu.reshape((z1 - z0, res, res))
            sdf_grid[z0:z1, :, :] = slab_sdfs_cpu

    # gaussian smoothing
    sigma = float(args.sigma)
    if sigma > 0:
        sdf_grid = gaussian_filter(sdf_grid, sigma=sigma)

    # marching cubes and export
    spacing = (float(z_vals_np[1] - z_vals_np[0]),
               float(y_vals_np[1] - y_vals_np[0]),
               float(x_vals_np[1] - x_vals_np[0]))

    mesh_o3d = marching_cubes_and_export(
        model, sdf_grid, level=0.0, spacing=spacing,
        min_bound=min_bound, output_mesh=args.output_mesh,
        device=device, vtx_batch=args.vtx_batch, amp_dtype=amp_dtype
    )

    # output slices
    res_mid = res // 2
    vmin, vmax = float(sdf_grid.min()), float(sdf_grid.max())
    save_slice_with_axes(sdf_grid[:, :, res_mid], args.output_mesh + "_slice_x.png", 'X', "SDF",
                         vmin=vmin, vmax=vmax, cmap='RdBu', colorbar_label='Signed Distance', show_sdf_zero=True)
    save_slice_with_axes(sdf_grid[:, res_mid, :], args.output_mesh + "_slice_y.png", 'Y', "SDF",
                         vmin=vmin, vmax=vmax, cmap='RdBu', colorbar_label='Signed Distance', show_sdf_zero=True)
    save_slice_with_axes(sdf_grid[res_mid, :, :], args.output_mesh + "_slice_z.png", 'Z', "SDF",
                         vmin=vmin, vmax=vmax, cmap='RdBu', colorbar_label='Signed Distance', show_sdf_zero=True)

    save_gradient_norm_slice(sdf_grid[:, res_mid, :], args.output_mesh + "_grdient_slice_y.png", 'Y', "SDF",
                         vmin=vmin, vmax=vmax, cmap='PuOr', colorbar_label='‖∇SDF‖', show_sdf_zero=True)

    print("Mesh and slices exported.")
