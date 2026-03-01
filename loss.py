import torch
import torch.nn.functional as F

# compute eik loss globally
def compute_global_eik_loss(model, x, ratio=1.5, expand=0.3):
    """
    x: [N,6] (點雲)
    ratio: 取樣數量 = ratio * 點雲數量
    expand: bounding box 擴張比例
    """
    device = x.device
    N = x.shape[0]
    N_eik = int(N * ratio)
    # 抽樣區間
    pts = x[:, :3]
    bbox_min, bbox_max = pts.min(0)[0], pts.max(0)[0]
    min_box = bbox_min - expand * (bbox_max - bbox_min)
    max_box = bbox_max + expand * (bbox_max - bbox_min)
    # 均勻隨機抽樣
    eik_xyz = torch.rand(N_eik, 3, device=device) * (max_box - min_box) + min_box
    # 拼成 (x, y, z, x, y, z)
    eik_x = torch.cat([eik_xyz], dim=1)
    eik_x.requires_grad_(True)
    sdf_pred, _ = model(eik_x)
    grad_outputs = torch.ones_like(sdf_pred, device=device)
    gradients = torch.autograd.grad(
        outputs=sdf_pred, inputs=eik_x, grad_outputs=grad_outputs,
        create_graph=True, retain_graph=True, only_inputs=True
    )[0][:, :3]
    eikonal_loss = ((gradients.norm(2, dim=-1) - 1) ** 2).mean()
    return eikonal_loss

def compute_negative_sdf_loss(
    model, 
    x, 
    ratio=0.5, 
    expand=0.5, 
    batch_size=2**12, 
    far_weight=8.0, 
    subsample_points=50000   # 可調：限制距離計算的對象點數
):
    device = x.device
    N_neg = int(x.shape[0] * ratio)

    # --- 建立 bounding box ---
    points = x[:, :3]
    bbox_min, bbox_max = points.min(0)[0], points.max(0)[0]
    min_neg = bbox_min - expand * (bbox_max - bbox_min)
    max_neg = bbox_max + expand * (bbox_max - bbox_min)

    # --- 隨機取樣 negative points ---
    neg_xyz = torch.rand(N_neg, 3, device=device) * (max_neg - min_neg) + min_neg

    # --- 距離計算 (子採樣避免爆記憶體) ---
    with torch.no_grad():
        if points.shape[0] > subsample_points:
            idx = torch.randperm(points.shape[0], device=device)[:subsample_points]
            ref_points = points[idx]
        else:
            ref_points = points

        # torch.cdist: [N_neg, M] -> min distance
        d = torch.cdist(neg_xyz, ref_points)
        neg_target, _ = d.min(dim=1)

    # --- loss 計算 ---
    far_threshold = 0.8 * (bbox_max - bbox_min).norm()
    losses = []

    for i in range(0, N_neg, batch_size):
        end = min(i + batch_size, N_neg)
        sdf_pred, _ = model(neg_xyz[i:end])  # (B,)
        gt = neg_target[i:end]

        mask_far = gt > far_threshold
        mask_near = ~mask_far

        loss = 0.0
        if mask_near.any():
            loss += F.l1_loss(sdf_pred[mask_near], gt[mask_near])
        if mask_far.any():
            loss += far_weight * F.mse_loss(sdf_pred[mask_far], gt[mask_far])
        losses.append(loss)

    return torch.stack(losses).mean()

def color_geometry_loss(x, f_x, alpha=10.0, sample_size=1024):
    # x: [N,7]; f_x: [N]
    N = x.shape[0]
    device = x.device
    idx = torch.randperm(N, device=device)[:sample_size]
    x_sub = x[idx]  # [sample_size, 6]
    f_sub = f_x[idx]  # [sample_size]
    rgb = x_sub[:, 3:]  # [sample_size, 3]

    # Pairwise color distance
    color_diff2 = (rgb.unsqueeze(1) - rgb.unsqueeze(0)).pow(2).sum(-1)  # [sample_size, sample_size]
    # Pairwise sdf distance
    sdf_diff2 = (f_sub.unsqueeze(1) - f_sub.unsqueeze(0)).pow(2)  # [sample_size, sample_size]
    # Weight: color越接近越重，越遠越輕
    w = torch.exp(-alpha * color_diff2)
    # 損失
    loss = (w * sdf_diff2).mean()
    return loss

# refer to SparseNeuS
def compute_sparse_loss(model, x, num_samples, box_margin=0.1, tau=30.0):
    with torch.no_grad():
        # 只取前三維 (xyz) 建立 bounding box
        min_bound = x[:, :3].min(dim=0)[0]
        max_bound = x[:, :3].max(dim=0)[0]
        margin = (max_bound - min_bound) * box_margin
        min_bound -= margin
        max_bound += margin

        # Uniform sample in expanded box
        uniform_3d = torch.rand((num_samples, 3), device=x.device) * (max_bound - min_bound) + min_bound

    sdf_pred, _ = model(uniform_3d)
    sparse_loss = torch.exp(-tau * sdf_pred.abs()).mean()
    return sparse_loss

def compute_normal_loss(model, x, normals, batch_size=8192):
    N = x.shape[0]
    total_loss = 0.0
    device = x.device

    for i in range(0, N, batch_size):
        x_batch = x[i:i+batch_size].clone().detach().requires_grad_(True)
        normals_batch = normals[i:i+batch_size]

        f_pred, _ = model(x_batch)
        grads = torch.autograd.grad(
            outputs=f_pred,
            inputs=x_batch,
            grad_outputs=torch.ones_like(f_pred),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0][:, :3]  # normal loss只對xyz，這裡選前3維

        # 只計算靠近表面的normal loss
        mask = (f_pred.abs() < 0.05)
        if mask.any():
            grads = grads[mask]
            normals_batch = normals_batch[mask]
            grads = F.normalize(grads, dim=1)
            normals_batch = F.normalize(normals_batch, dim=1)
            cos_sim = F.cosine_similarity(grads, normals_batch, dim=1)
            loss = ((1 - cos_sim) ** 2).mean()
            total_loss += loss * len(grads)
    return total_loss / N

# total loss calculation
def compute_loss(
        model, 
        x,                # [N, 3]
        gt_colors,
        x_noisy_full,     # [N, 3]
        epsilon,          # [N, 3]
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
        weight_singular_hessian,
        weight_rgb_gt
    ):

    batch_size = 2**18 if is_a100 else 30000
    # forward cache（避免重複 forward）
    sdf_noisy, _ = model(x_noisy_full)    # noisy input (SDF loss 用)
    sdf_x, rgb_x = model(x)               # 原始點 (zero / rgb / color geo 用)

    # 1) SDF Loss
    if weight_sdf > 0.0:
        if not isinstance(normals, torch.Tensor):
            normals_t = torch.tensor(normals, dtype=torch.float32, device=x.device)
        else:
            normals_t = normals
        signed_dist = torch.sum(epsilon[:, :3] * normals_t, dim=1)
        loss_sdf = F.mse_loss(sdf_noisy, signed_dist)
    else:
        loss_sdf = torch.tensor(0.0, device=x.device)

    # 2) Singular Hessian（需額外建圖，保留原做法）
    if weight_singular_hessian > 0.0 and epoch % 10 == 0:
        with torch.no_grad():
            sdf_vals = sdf_noisy  # 已經有前向結果
            surface_mask = sdf_vals.abs() < 0.1
            x_near_surface = x_noisy_full[surface_mask].detach()

        if x_near_surface.shape[0] < 8:
            loss_singular_hessian = torch.tensor(0.0, device=x.device)
        else:
            loss_list = []
            for start in range(0, x_near_surface.shape[0], batch_size):
                end = min(start + batch_size, x_near_surface.shape[0])
                x_batch_full = x_near_surface[start:end]
                x_batch_xyz = x_batch_full[:, :3].detach().clone().requires_grad_(True)
                x_batch_extra = x_batch_full[:, 3:]
                x_input = torch.cat([x_batch_xyz, x_batch_extra], dim=1)

                sdf_pred_b, _ = model(x_input)

                gradients = torch.autograd.grad(
                    outputs=sdf_pred_b.sum(),        # ← 等價於 ones_like
                    inputs=x_batch_xyz,
                    create_graph=True,
                    retain_graph=False,              # ← 不共享圖，無須保留
                    only_inputs=True
                )[0]  # [B, 3]

                hessian_rows = []
                for i in range(3):
                    grad_i = gradients[:, i]
                    grad2 = torch.autograd.grad(
                        outputs=grad_i,
                        inputs=x_batch_xyz,
                        grad_outputs=torch.ones_like(grad_i),
                        create_graph=False,
                        retain_graph=(i < 2),
                        only_inputs=True,
                        allow_unused=True
                    )[0]
                    if grad2 is None:
                        grad2 = torch.zeros_like(x_batch_xyz)
                    hessian_rows.append(grad2.unsqueeze(1))
                hessian = torch.cat(hessian_rows, dim=1)  # [B, 3, 3]
                eigvals = torch.linalg.eigvalsh(hessian)
                eig_max = eigvals.abs().max(dim=1).values
                loss_list.append(eig_max.mean())

            loss_singular_hessian = torch.stack(loss_list).mean()
    else:
        loss_singular_hessian = torch.tensor(0.0, device=x.device)

    # 3) Zero loss（直接用 cache）
    loss_zero = sdf_x.abs().mean() if weight_zero > 0.0 else torch.tensor(0.0, device=x.device)

    # 4) Eikonal（two-pass：各自建圖，不共享）
    if eik_init > 0.0 and weight_glob_eik > 0.0:
        with torch.no_grad():
            idx = torch.where(sdf_noisy.abs().squeeze(-1) < 0.2)[0]
            if idx.numel() > 30000:
                idx = idx[torch.randperm(idx.numel(), device=idx.device)[:30000]]
        if idx.numel() > 0:
            x_eik = x_noisy_full[idx].detach().clone().requires_grad_(True)
            f_sub, _ = model(x_eik)
            grads = torch.autograd.grad(outputs=f_sub.sum(), inputs=x_eik, create_graph=True)[0][:, :3]
            loss_eikonal_surface = ((grads.norm(dim=-1) - 1) ** 2).mean()
            if epoch < 0.8 * total_epochs:
                loss_eikonal_global = compute_global_eik_loss(model, x, ratio=0.6 if is_a100 else 1.2, expand=0.7)
            else:
                loss_eikonal_global = torch.tensor(0.0, device=x.device)
        else:
            loss_eikonal_surface = torch.tensor(0.0, device=x.device)
            loss_eikonal_global  = torch.tensor(0.0, device=x.device)
    else:
        loss_eikonal_surface = torch.tensor(0.0, device=x.device)
        loss_eikonal_global  = torch.tensor(0.0, device=x.device)

    # 5) Normal loss（需 requires_grad）
    if weight_normal > 0.0:
        if not isinstance(normals, torch.Tensor):
            normals_t = torch.tensor(normals, dtype=torch.float32, device=x.device)
        else:
            normals_t = normals
        loss_normal = compute_normal_loss(model, x, normals_t, batch_size=batch_size)
        if not isinstance(loss_normal, torch.Tensor):
            loss_normal = torch.tensor(loss_normal, device=x.device)
    else:
        loss_normal = torch.tensor(0.0, device=x.device)

    # 6) Sparse loss（需各自 forward）
    loss_sparse = compute_sparse_loss(model, x, num_samples=batch_size) if weight_sparse > 0.0 else torch.tensor(0.0, device=x.device)

    # 7) Color-Geometry（重用 sdf_x）
    loss_color_geo = color_geometry_loss(x, sdf_x, alpha=10.0, sample_size=batch_size) if weight_color_geo > 0.0 else torch.tensor(0.0, device=x.device)

    # 8) Negative SDF（內部各自 forward）
    loss_neg_sdf = compute_negative_sdf_loss(model, x, ratio=0.05, expand=0.5, batch_size=batch_size) if weight_neg_sdf > 0.0 else torch.tensor(0.0, device=x.device)

    # 9) RGB GT（重用 rgb_x）
    loss_rgb_gt = F.mse_loss(rgb_x, gt_colors) if weight_rgb_gt > 0.0 else torch.tensor(0.0, device=x.device)

    return (loss_sdf, loss_zero, loss_eikonal_surface, 
            loss_eikonal_global, loss_normal, loss_sparse, 
            loss_color_geo, loss_neg_sdf, loss_singular_hessian, loss_rgb_gt
        )
