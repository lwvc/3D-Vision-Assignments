# edit_object_extract.py
# 基於 edit_object_removal.py，改為「只保留」目標物件

import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, OptimizationParams, get_combined_args
from gaussian_renderer import GaussianModel
import numpy as np
from PIL import Image
import json
import cv2
from scipy.spatial import Delaunay
from render import feature_to_rgb, visualize_obj

# 直接從 edit_object_removal.py 複製這個函式，完全不變
def points_inside_convex_hull(point_cloud, mask, remove_outliers=True, outlier_factor=1.0):
    masked_points = point_cloud[mask].cpu().numpy()
    if remove_outliers:
        Q1 = np.percentile(masked_points, 25, axis=0)
        Q3 = np.percentile(masked_points, 75, axis=0)
        IQR = Q3 - Q1
        outlier_mask = (masked_points < (Q1 - outlier_factor * IQR)) | (masked_points > (Q3 + outlier_factor * IQR))
        filtered_masked_points = masked_points[~np.any(outlier_mask, axis=1)]
    else:
        filtered_masked_points = masked_points
    delaunay = Delaunay(filtered_masked_points)
    points_inside_hull_mask = delaunay.find_simplex(point_cloud.cpu().numpy()) >= 0
    inside_hull_tensor_mask = torch.tensor(points_inside_hull_mask, device='cuda')
    return inside_hull_tensor_mask


def extract_setup(opt, model_path, iteration, gaussians, classifier, selected_obj_ids, removal_thresh, outlier_factor):
    selected_obj_ids = torch.tensor(selected_obj_ids).cuda()

    with torch.no_grad():
        logits3d = classifier(gaussians._objects_dc.permute(2,0,1))
        prob_obj3d = torch.softmax(logits3d, dim=0)
        mask = prob_obj3d[selected_obj_ids, :, :] > removal_thresh
        mask3d = mask.any(dim=0).squeeze()

        print(f"[INFO] 分類器找到的 Gaussians 數量: {mask3d.sum().item()}")

        # Convex Hull 擴展（補抓被遮擋的內部 Gaussians）
        mask3d_convex = points_inside_convex_hull(
            gaussians._xyz.detach(), mask3d, outlier_factor=outlier_factor
        )
        mask3d = torch.logical_or(mask3d, mask3d_convex)

        print(f"[INFO] Convex Hull 擴展後 Gaussians 數量: {mask3d.sum().item()}")
        print(f"[INFO] 總 Gaussians 數量: {mask3d.shape[0]}")

    # ===== 關鍵差異 =====
    # removal 是把 mask3d=True 的點刪掉（opacity→0）
    # extract 是把 mask3d=False 的點刪掉，只保留目標
    keep_mask = mask3d  # True = 保留（熊），False = 丟棄（背景）

    # 直接過濾所有 Gaussian 屬性，只保留目標物件
    gaussians._xyz = gaussians._xyz[keep_mask]
    gaussians._features_dc = gaussians._features_dc[keep_mask]
    gaussians._features_rest = gaussians._features_rest[keep_mask]
    gaussians._scaling = gaussians._scaling[keep_mask]
    gaussians._rotation = gaussians._rotation[keep_mask]
    gaussians._opacity = gaussians._opacity[keep_mask]
    gaussians._objects_dc = gaussians._objects_dc[keep_mask]

    print(f"[INFO] 提取完成，保留 {gaussians._xyz.shape[0]} 個 Gaussians")

    # 儲存
    point_cloud_path = os.path.join(
        model_path,
        "point_cloud_object_extract/iteration_{}".format(iteration)
    )
    os.makedirs(point_cloud_path, exist_ok=True)
    save_path = os.path.join(point_cloud_path, "point_cloud.ply")
    gaussians.save_ply(save_path)
    print(f"[INFO] 已儲存到: {save_path}")

    return gaussians


def extract(dataset: ModelParams, iteration: int, pipeline: PipelineParams,
            opt: OptimizationParams, select_obj_id: int,
            removal_thresh: float, outlier_factor: float):

    # 載入訓練好的模型
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

    num_classes = dataset.num_classes
    print("Num classes:", num_classes)

    classifier = torch.nn.Conv2d(gaussians.num_objects, num_classes, kernel_size=1)
    classifier.cuda()
    classifier.load_state_dict(torch.load(
        os.path.join(dataset.model_path, "point_cloud",
                     "iteration_" + str(scene.loaded_iter), "classifier.pth")
    ))

    bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    # 執行提取
    extract_setup(opt, dataset.model_path, scene.loaded_iter,
                  gaussians, classifier, select_obj_id,
                  removal_thresh, outlier_factor)


if __name__ == "__main__":
    parser = ArgumentParser(description="Object extraction script")
    model = ModelParams(parser, sentinel=True)
    opt = OptimizationParams(parser)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--config_file", type=str,
                        default="config/object_removal/bear_outlier10.json",
                        help="Path to the configuration file")

    args = get_combined_args(parser)
    print("Extracting from " + args.model_path)

    # 讀 config（和 removal 用同一個 json 即可）
    try:
        with open(args.config_file, 'r') as file:
            config = json.load(file)
    except FileNotFoundError:
        print(f"Error: Configuration file '{args.config_file}' not found.")
        exit(1)

    args.num_classes = config.get("num_classes", 256)
    args.removal_thresh = config.get("removal_thresh", 0.3)
    args.select_obj_id = config.get("select_obj_id", [34])
    args.outlier_factor = config.get("outlier_factor", 1.0)

    safe_state(args.quiet)

    extract(model.extract(args), args.iteration, pipeline.extract(args),
            opt.extract(args), args.select_obj_id,
            args.removal_thresh, args.outlier_factor)
