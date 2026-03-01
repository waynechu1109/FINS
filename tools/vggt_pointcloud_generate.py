import sys
import argparse
# sys.path.append('./../vggt')
sys.path.append('deps/vggt')

from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map
import torch
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
# from skimage.filters import threshold_otsu
from scipy.spatial import cKDTree
from time import time

import matplotlib.pyplot as plt
import open3d as o3d
import numpy as np
import os 
import glob
import json

parser = argparse.ArgumentParser(description='VGGT inference and point cloud downsampling')
# parser.add_argument('--downsample', type=str, default='true',
#                     help="downsample the pointcloud or not.")
parser.add_argument('--max_points', type=int, default=200000,
                    help='the max number of points after downsample')
parser.add_argument('--voxel_size', type=float, default=0.00001,
                    help='Voxel Grid Size')
parser.add_argument('--file', type=str, help="File name to render.")
parser.add_argument('--thres', type=float, default=0, help="Theshold.")
parser.add_argument('--concave', type=str, default='false',
                    help="The scene is concave or not.")
# parser.add_argument('--plane_rm', type=str, default='false',
#                     help="Remove plane or not.")
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
# bfloat16 is supported on Ampere GPUs (Compute Capability 8.0+) 
dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

# Initialize the model and load the pretrained weights.
# This will automatically download the model weights the first time it's run, which may take a while.
# model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)

total_start = time()

# If the model is downloaded on the device
load_model_start = time()
model = VGGT()
model.load_state_dict(torch.load("deps/vggt/model/model.pt", map_location=device))
load_model_end = time()

model.to(device)

file_name = args.file
max_points = args.max_points
# downsample = args.downsample
thres = args.thres
concave = args.concave
# plane_rm = args.plane_rm
if concave != 'true' and concave != 'false':
    print('Incorrect arguments, only true or false is accepted.')
    sys.exit(1)
# if plane_rm != 'true' and plane_rm != 'false':
#     print('Incorrect arguments, only true or false is accepted.')
#     sys.exit(1)

image_candidates = []

target_path = os.path.join("data", file_name)

load_img_start = time()
if os.path.isdir(target_path):
    image_candidates = sorted(
        glob.glob(os.path.join(target_path, "*.jpg")) +
        glob.glob(os.path.join(target_path, "*.png")) +
        glob.glob(os.path.join(target_path, "*.JPG")) +
        glob.glob(os.path.join(target_path, "*.PNG"))
    )
    if len(image_candidates) == 0:
        raise FileNotFoundError(f"No images found in directory {target_path}")
else:
    # Assume file_name is base name of an image without extension
    for ext in ["jpg", "png", "JPG", "PNG"]:
        candidate = os.path.join("data", f"{file_name}.{ext}")
        if os.path.isfile(candidate):
            image_candidates = [candidate]
            break
    if len(image_candidates) == 0:
        raise FileNotFoundError(f"No image found for {file_name}.jpg or .png in ../data/")

print(f"Found {len(image_candidates)} image(s).")

images_list = image_candidates
images = load_and_preprocess_images(images_list).to(device)
load_img_end = time()

# with torch.no_grad():
#     with torch.cuda.amp.autocast(dtype=dtype):
#         # Predict attributes including cameras, depth maps, and point maps.
#         predictions = model(images)


with torch.no_grad():
    with torch.cuda.amp.autocast(dtype=dtype):
        images = images[None]  # add batch dimension
        aggregated_tokens_list, ps_idx = model.aggregator(images)
                
    # Predict Cameras
    camera_start = time()
    pose_enc = model.camera_head(aggregated_tokens_list)[-1]
    # Extrinsic and intrinsic matrices, following OpenCV convention (camera from world)
    extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])
    camera_end = time()
    print(f"VGGT input resolution (HxW): {images.shape[-2:]}")
    # print(f'extrinsic: {extrinsic}, intrinsic: {intrinsic}')

    # Predict Depth Maps
    depth_start = time()
    depth_map, depth_conf = model.depth_head(aggregated_tokens_list, images, ps_idx)
    depth_end = time()

    # Predict Point Maps
    point_start = time()
    point_map, point_conf = model.point_head(aggregated_tokens_list, images, ps_idx)
    point_end = time()
        
    # Construct 3D Points from Depth Maps and Cameras
    # which usually leads to more accurate 3D points than point map branch
    point_map_by_unprojection = unproject_depth_map_to_point_map(depth_map.squeeze(0), 
                                                                extrinsic.squeeze(0), 
                                                                intrinsic.squeeze(0))

    # Predict Tracks
    # choose your own points to track, with shape (N, 2) for one scene
    query_points = torch.FloatTensor([[100.0, 200.0], 
                                        [60.72, 259.94]]).to(device)
    track_list, vis_score, conf_score = model.track_head(aggregated_tokens_list, images, ps_idx, query_points=query_points[None])

    # print(f'depth_conf: {depth_conf}, shape: {depth_conf.shape}')
    # print(f'point_conf: {point_conf}, shape: {point_conf.shape}')
    # print(f'conf_score: {conf_score}, shape: {conf_score.shape}')

    # 準備最終點雲與顏色
    all_pts3d = []
    all_colors = []

    all_confidence = []

    B = point_map_by_unprojection.shape[0]  # batch size
    for b in range(B):
        points = point_map_by_unprojection[b]  # [H, W, 3]
        H, W, _ = points.shape
        points = points.reshape(-1, 3)         # ➜ [H*W, 3]

        # Depth confidence 對應影像 b
        depth_conf_cpu = depth_conf[0, b].detach().cpu().numpy()  # [H, W]
        depth_conf_flat = depth_conf_cpu.reshape(-1)              # [H*W]

        if thres:
            # 門檻
            threshold = np.percentile(depth_conf_flat, thres)
            conf_mask = depth_conf_flat >= threshold

            # 有效點過濾
            valid_mask = np.isfinite(points).all(axis=1) & (np.abs(points).sum(axis=1) > 0)
            final_mask = valid_mask & conf_mask
        else:
            final_mask = np.isfinite(points).all(axis=1)  # 只排除 NaN/Inf

        # 過濾點雲
        filtered_points = points[final_mask]
        all_pts3d.append(filtered_points)

        # 處理單張圖片的 RGB 張量
        img_rgb = images[0, b]  # [3, H, W]
        if isinstance(img_rgb, torch.Tensor):
            img_rgb = img_rgb.detach().cpu().numpy()

        if img_rgb.ndim == 3 and img_rgb.shape[0] == 3:
            img_rgb = np.transpose(img_rgb, (1, 2, 0))  # ➜ [H, W, 3]
        elif img_rgb.ndim == 2:
            img_rgb = np.stack([img_rgb]*3, axis=-1)
        else:
            raise ValueError(f"Unexpected image shape after indexing: {img_rgb.shape}")

        img_rgb = img_rgb.reshape(-1, 3)
        img_rgb = img_rgb[final_mask]
        all_colors.append(np.clip(img_rgb, 0, 1))

        conf_values = depth_conf_flat  # ← 這行要加上，否則你在 all_confidence.append(conf_values[...]) 時會報錯
        all_confidence.append(conf_values[final_mask])  # 因用 final_mask 做過濾
        

    # 合併所有圖片的點雲, 顏色, confidence
    all_pts3d = np.concatenate(all_pts3d, axis=0)
    all_colors = np.concatenate(all_colors, axis=0)
    all_confidence = np.concatenate(all_confidence, axis=0)  # 假設你前面已記錄每張影像的 confidence 值    

    # # Normalize to [-1, 1]
    # mins = all_pts3d.min(axis=0)
    # maxs = all_pts3d.max(axis=0)
    # centre = (maxs + mins) / 2.0
    # scale = np.max(maxs - mins) / 2.0
    # all_pts3d = (all_pts3d - centre) / scale

    # Normalize to [0, 1] for hash encoding
    mins = all_pts3d.min(axis=0)         # shape (3,)
    maxs = all_pts3d.max(axis=0)         # shape (3,)
    centre = (maxs + mins) / 2.0
    scale = (maxs - mins).max()          # 取最大邊長
    all_pts3d = (all_pts3d - centre) / scale + 0.5

    # 建立 Open3D 點雲
    pcd = o3d.geometry.PointCloud()

    if max_points:
        N = all_pts3d.shape[0]
        if N > args.max_points:
            idx = np.random.choice(N, size=args.max_points, replace=False)
            all_pts3d = all_pts3d[idx]
            all_colors = all_colors[idx]
            all_confidence = all_confidence[idx]
        pcd.points = o3d.utility.Vector3dVector(all_pts3d)
        # pcd.colors = o3d.utility.Vector3dVector(all_colors.astype(float))
        pcd.colors = o3d.utility.Vector3dVector(all_colors.astype(float))
        pcd_down = pcd.voxel_down_sample(voxel_size=args.voxel_size)
    else: 
        pcd.points = o3d.utility.Vector3dVector(all_pts3d)
        # pcd.colors = o3d.utility.Vector3dVector(all_colors.astype(float))
        pcd.colors = o3d.utility.Vector3dVector(all_colors.astype(float))
        pcd_down = pcd

    # 法線估計與方向一致化
    pcd_down.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30))
    pcd_down.orient_normals_consistent_tangent_plane(k=30)

    if concave == 'true':
        # reverse the normals if in "concave" scene
        pcd_down.normals = o3d.utility.Vector3dVector(-np.asarray(pcd_down.normals))

    # 離群點移除
    pcd_down, ind = pcd_down.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

    # ======= RANSAC plane removal =======
    # if plane_rm == 'true':
    #     distance_threshold = 0.01
    #     ransac_n = 3
    #     num_iterations = 1000

    #     plane_model, inliers = pcd_down.segment_plane(distance_threshold=distance_threshold,
    #                                                 ransac_n=ransac_n,
    #                                                 num_iterations=num_iterations)
    #     print(f"Detected plane equation: {plane_model[0]:.3f}x + {plane_model[1]:.3f}y + {plane_model[2]:.3f}z + {plane_model[3]:.3f} = 0")

    #     # 移除最大平面（通常為桌面/背景）
    #     pcd_down = pcd_down.select_by_index(inliers, invert=True)

    # 使用 KDTree 對應最終點雲與原始 confidence 值
    tree = cKDTree(all_pts3d)
    final_points = np.asarray(pcd_down.points)

    # 查詢最接近的原始點 index
    _, indices = tree.query(final_points, k=1)
    output_conf = [float(all_confidence[i]) for i in indices]

    # 匯出點雲
    os.makedirs(f"data/vggt_preprocessed/{file_name}", exist_ok=True)
    o3d.io.write_point_cloud(f"data/vggt_preprocessed/{file_name}/output_pointcloud_normal.ply", pcd_down)
    json_path = f"data/vggt_preprocessed/{file_name}/output_confidence.json"
    with open(json_path, 'w') as f:
        json.dump(output_conf, f, indent=2)
    # store the centre point and scale information for inferencing 
    with open(f"data/vggt_preprocessed/{file_name}/output_pointcloud_normal_info.json", "w") as f:
        json.dump({
            "centre": centre.tolist(), 
            "scale": scale,
            "extrinsic": extrinsic.tolist(),
            "intrinsic": intrinsic.tolist()
        }, f, indent=4, sort_keys=True)

total_end = time()

# print time
print(f"Load model took {load_model_end - load_model_start:.2f} seconds.")
print(f"Load image took {load_img_end - load_img_start:.2f} seconds.")
print(f"Camera pose prediction took {camera_end - camera_start:.2f} seconds.")
print(f"Depth prediction took {depth_end - depth_start:.2f} seconds.")
print(f"Point map prediction took {point_end - point_start:.2f} seconds.")

print(f"Total processing time: {total_end - total_start:.2f} seconds.")


#### debugging image output ####

# === 讀取資料 ===
# info_path = f"../data/vggt_preprocessed/{file_name}/output_pointcloud_normal_info.json"
# pcd_path = f"../data/vggt_preprocessed/{file_name}/output_pointcloud_normal.ply"

# with open(info_path, 'r') as f:
#     info = json.load(f)

# centre = np.array(info['centre'])
# scale = info['scale']
# extrinsic = np.array(info['extrinsic']).reshape(3, 4)
# intrinsic = np.array(info['intrinsic']).reshape(3, 3)

# # 讀取點雲
# pcd = o3d.io.read_point_cloud(pcd_path)
# points = np.asarray(pcd.points) * scale + centre  # 還原原始點
# colors = np.asarray(pcd.colors)  # RGB 顏色，shape = (N, 3)
# # points = np.asarray(pcd.points)
# N = points.shape[0]
# points_hom = np.hstack([points, np.ones((N, 1))]).T  # (4, N)

# # 投影到像素空間
# cam_points = extrinsic @ points_hom  # (4, N)
# cam_points = cam_points[:3, :]       # (3, N)
# pixels = intrinsic @ cam_points      # (3, N)
# pixels = pixels / pixels[2:3, :]     # 除以深度 Z
# pixel_coords = pixels[:2, :].T       # (N, 2)

# # 設定影像大小（你可以依你的資料調整）
# image_width = 1024
# image_height = 1024

# # 過濾合法像素位置
# x, y = pixel_coords[:, 0], pixel_coords[:, 1]
# valid_mask = (x >= 0) & (x < image_width) & (y >= 0) & (y < image_height)
# valid_pixel_coords = pixel_coords[np.where(valid_mask)[0]]
# valid_colors = colors[np.where(valid_mask)[0]]

# # === 繪圖 ===
# plt.figure(figsize=(10, 10))
# plt.scatter(valid_pixel_coords[:, 0], valid_pixel_coords[:, 1],
#             c=valid_colors, s=1.5, alpha=0.8)  # 可以改點的顏色與大小
# plt.gca().invert_yaxis()  # 類似 OpenCV 的座標系統：原點在左上
# plt.axis('off')
# plt.xlim([0, image_width])
# plt.ylim([image_height, 0])
# plt.tight_layout()
# plt.savefig(f"../data/vggt_preprocessed/{file_name}/debug_cloud_only.png", dpi=300)

