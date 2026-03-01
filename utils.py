import torch
from skimage import measure
import numpy as np
from tqdm import tqdm
import open3d as o3d

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('true', '1'):
        return True
    elif v.lower() in ('false', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def render_pointcloud(points: torch.Tensor,
                      colors: torch.Tensor,
                      cam_pose: torch.Tensor,
                      K: torch.Tensor,
                      image_size: tuple[int, int]
                      ) -> tuple[torch.Tensor, torch.Tensor]:
    device = points.device
    H, W = image_size

    n = points.shape[0]
    pts_h = torch.cat([points, torch.ones(n, 1, device=device)], dim=1)  # (n,4)
    cam_h = (cam_pose @ pts_h.T).T
    Xc, Yc, Zc = cam_h[:, 0], cam_h[:, 1], cam_h[:, 2]

    proj = (K @ cam_h[:, :3].T).T
    u = (proj[:, 0] / proj[:, 2]).round().long()
    v = (proj[:, 1] / proj[:, 2]).round().long()

    valid = (Zc > 0) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
    u, v, Zc, cols = u[valid], v[valid], Zc[valid], colors[valid]

    image = torch.zeros((H, W, 3), device=device)
    depth = torch.full((H, W), float('inf'), device=device)

    for xi, yi, zi, ci in zip(u, v, Zc, cols):
        if zi < depth[yi, xi]:
            depth[yi, xi] = zi
            image[yi, xi] = ci

    return image, depth

def marching_cubes_and_export(model, sdf_grid, level, spacing, min_bound, output_mesh, device, vtx_batch=1<<20, amp_dtype=None):
    # Marching Cubes（CPU）
    verts, faces, normals, _ = measure.marching_cubes(sdf_grid, level=level, spacing=spacing)
    # 轉換座標系
    verts = verts + np.array(min_bound)[::-1]
    verts = verts[:, [2, 1, 0]]
    # 頂點上色（GPU，分批）
    verts_colors = np.zeros((verts.shape[0], 3), dtype=np.float32)

    if amp_dtype is None:
        amp_ctx = torch.cuda.amp.autocast(enabled=False)
    else:
        amp_ctx = torch.cuda.amp.autocast(dtype=amp_dtype)

    with torch.inference_mode(), amp_ctx:
        for i in tqdm(range(0, verts.shape[0], vtx_batch), desc="Colorizing vertices"):
            j = min(i + vtx_batch, verts.shape[0])
            v_chunk = torch.as_tensor(verts[i:j], dtype=torch.float32, device=device)
            # 你的 SDFNet 接口: f(xyz)->(sdf, rgb)
            _, rgb = model(v_chunk)
            verts_colors[i:j] = rgb.clamp(0.0, 1.0).float().cpu().numpy()

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts)
    mesh.triangles = o3d.utility.Vector3iVector(faces.astype(np.int32))
    mesh.vertex_colors = o3d.utility.Vector3dVector(verts_colors)
    mesh.compute_vertex_normals()

    safe_level = str(level).replace(".", "p")
    output_path = f"{output_mesh}_sdf{safe_level}.ply"
    o3d.io.write_triangle_mesh(output_path, mesh)
    print(f"Exported colored mesh: {output_path}")
    return mesh

def save_slice_with_axes(
    img,
    path,
    axis_label,
    title,
    vmin,
    vmax,
    cmap,
    colorbar_label,
    scale=1.0,
    show_sdf_zero=False
):
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    # --- Step 1: 找 0-level set 範圍 ---
    zero_mask = np.isclose(img, 0.0, atol=(vmax - vmin) * 0.01)
    if not zero_mask.any():
        y_min, y_max = 0, img.shape[0]
        x_min, x_max = 0, img.shape[1]
    else:
        ys, xs = np.where(zero_mask)
        y_min, y_max = ys.min(), ys.max()
        x_min, x_max = xs.min(), xs.max()

        # --- Step 2: 往外推 10% ---
        h, w = img.shape
        pad_y = int(0.1 * (y_max - y_min))
        pad_x = int(0.1 * (x_max - x_min))

        y_min = max(y_min - pad_y, 0)
        y_max = min(y_max + pad_y, h)
        x_min = max(x_min - pad_x, 0)
        x_max = min(x_max + pad_x, w)

        # --- Step 3: 強制正方形 ---
        box_h = y_max - y_min
        box_w = x_max - x_min
        box_size = max(box_h, box_w)

        cy = (y_min + y_max) // 2
        cx = (x_min + x_max) // 2
        half = box_size // 2

        y_min = max(cy - half, 0)
        y_max = min(cy + half, h)
        x_min = max(cx - half, 0)
        x_max = min(cx + half, w)

    # --- Step 4: 裁切 ---
    img_crop = img[y_min:y_max, x_min:x_max]

    # --- Step 5: 繪圖 ---
    fig, ax = plt.subplots(figsize=(6, 6))  # square figure
    im = ax.imshow(
        img_crop, cmap=cmap, vmin=vmin, vmax=vmax, origin='lower',
        extent=[x_min, x_max, y_min, y_max]
    )

    contour_levels = np.linspace(vmin, vmax, 21)
    contours = ax.contour(
        img_crop, levels=contour_levels, colors='black', linewidths=1.2,
        origin='lower', extent=[x_min, x_max, y_min, y_max]
    )
    fmt = lambda x: f"{x*scale:.2f}"
    # ax.clabel(contours, fmt=fmt, fontsize=7)

    if show_sdf_zero:
        ax.contour(
            img_crop, levels=[0.0], colors='lime', linewidths=2, origin='lower',
            extent=[x_min, x_max, y_min, y_max]
        )

    # ax.set_title(f'{title} Slice - {axis_label} axis')

    ax.set_xticks([])
    ax.set_yticks([])

    # --- Step 6: colorbar 與圖片同高 ---
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label(f"{colorbar_label} (normalized)")

    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()

def save_gradient_norm_slice(
    img,
    path,
    axis_label,
    title,
    vmin,
    vmax,
    cmap="jet",
    colorbar_label="‖∇SDF‖",
    scale=1.0,
    show_sdf_zero=False
):
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    # --- Step 1: 計算 gradient norm ---
    gy, gx = np.gradient(img)
    grad_norm = np.sqrt(gx**2 + gy**2)

    # --- Step 2: 找 0-level set 範圍（還是基於 SDF 值） ---
    zero_mask = np.isclose(img, 0.0, atol=(vmax - vmin) * 0.01)
    if not zero_mask.any():
        y_min, y_max = 0, img.shape[0]
        x_min, x_max = 0, img.shape[1]
    else:
        ys, xs = np.where(zero_mask)
        y_min, y_max = ys.min(), ys.max()
        x_min, x_max = xs.min(), xs.max()

        h, w = img.shape
        pad_y = int(0.1 * (y_max - y_min))
        pad_x = int(0.1 * (x_max - x_min))

        y_min = max(y_min - pad_y, 0)
        y_max = min(y_max + pad_y, h)
        x_min = max(x_min - pad_x, 0)
        x_max = min(x_max + pad_x, w)

        # 強制正方形
        box_h = y_max - y_min
        box_w = x_max - x_min
        box_size = max(box_h, box_w)

        cy = (y_min + y_max) // 2
        cx = (x_min + x_max) // 2
        half = box_size // 2

        y_min = max(cy - half, 0)
        y_max = min(cy + half, h)
        x_min = max(cx - half, 0)
        x_max = min(cx + half, w)
    contour_levels = np.linspace(vmin, vmax, 21)

    # --- Step 3: 裁切 ---
    grad_crop = grad_norm[y_min:y_max, x_min:x_max]
    img_crop = img[y_min:y_max, x_min:x_max]

    # --- Step 4: 設定 gmin/gmax ---
    gmin = float(np.min(grad_crop))
    gmax = float(np.max(grad_crop))

    # --- Step 5: 繪圖 ---
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(
        grad_crop, cmap=cmap, vmin=gmin, vmax=gmax, origin='lower',
        extent=[x_min, x_max, y_min, y_max]
    )
    contours = ax.contour(
        img_crop, levels=contour_levels, colors='black', linewidths=1.2,
        origin='lower', extent=[x_min, x_max, y_min, y_max]
    )
    if show_sdf_zero:
        ax.contour(
            img_crop, levels=[0.0], colors='lime', linewidths=2, origin='lower',
            extent=[x_min, x_max, y_min, y_max]
        )

    ax.set_xticks([])
    ax.set_yticks([])

    # colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label(f"{colorbar_label}")

    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def build_coords_slab_on_device(x_vals, y_vals, z_vals, z_start, z_end, device):
    """
    在 GPU 端建立 [z_start:z_end) 這個 slab 的 (res_y*res_x*(z_end-z_start), 3) 座標
    """
    # 注意：x_vals/y_vals/z_vals 已是 torch.Tensor 且在 device 上
    z_chunk = z_vals[z_start:z_end]                       # (Dz,)
    yy, xx = torch.meshgrid(y_vals, x_vals, indexing='ij')# (Ry,Rx)
    # 展平 YX -> (Ry*Rx, )
    xy = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=-1) # (Ry*Rx,2)
    # 重複到每個 Z 切片
    xy = xy.unsqueeze(0).expand(z_chunk.shape[0], -1, -1)     # (Dz, Ry*Rx, 2)
    zz = z_chunk[:, None, None].expand(-1, xy.shape[1], 1)    # (Dz, Ry*Rx, 1)
    pts = torch.cat([xy, zz], dim=-1).reshape(-1, 3)          # (Dz*Ry*Rx, 3) = (slab_size, 3)
    return pts