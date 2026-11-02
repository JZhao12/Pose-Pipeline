#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 11:25:27 2026

@author: Jianyu Zhao
"""

import argparse
import logging
import os
import torch
import sys
import cv2
import numpy as np
import imageio.v2 as imageio
import trimesh
from VOT import Cutie
from utils.kalman_filter_6d import KalmanFilter6D
from utils.pose_utils import (
    LOST_BBOX, mat_to_pose6, pose6_to_mat, is_valid_bbox, is_plausible_pose,
    bbox_center, image_point_to_xy,
    read_depth_m, read_mask, check_sequence_dir,
)


src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
foundationpose_path = os.path.join(src_path, "FoundationPose")
if src_path not in sys.path:
    sys.path.append(src_path)
if foundationpose_path not in sys.path:
    sys.path.append(foundationpose_path)


def pose_track(
        video_dir: str,
        results_dir: str,
        est_refine_iter: int,
        track_refine_iter: int,
        kf_measurement_noise_scale: float,
        show_vis: bool = True,
):
    if show_vis and sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
        logging.warning("未检测到 DISPLAY，自动关闭 cv2 窗口（等价于 --no_vis）")
        show_vis = False
    check_sequence_dir(video_dir)
    from FoundationPose.datareader import YcbineoatReader
    reader = YcbineoatReader(video_dir=video_dir, shorter_side=None, zfar=np.inf)
    cam_K = reader.K
    depth_scale = np.loadtxt(f'{video_dir}/depth_scale.txt')
    total_frames = len(reader)

    pose_output_path = os.path.join(results_dir, "pose.npy")
    mask_visualization_path = os.path.join(results_dir, "mask_visualization")
    bbox_visualization_path = os.path.join(results_dir, "bbox_visualization")
    pose_visualization_path = os.path.join(results_dir, "pose_visualization")

    os.makedirs(mask_visualization_path, exist_ok=True)
    os.makedirs(bbox_visualization_path, exist_ok=True)
    os.makedirs(pose_visualization_path, exist_ok=True)

    mesh_file = os.path.join(video_dir, "mesh", "mesh.obj")
    mesh = trimesh.load(mesh_file, force="mesh")
    if float(np.max(mesh.extents)) > 5.0:
        logging.warning(f"mesh 最大边长 {np.max(mesh.extents):.1f}，疑似毫米单位；流水线要求米")

    to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
    bbox = np.stack([-extents / 2, extents / 2], axis=0).reshape(2, 3)

    from FoundationPose.estimater import (
        ScorePredictor, PoseRefinePredictor, dr,
        FoundationPose, draw_posed_3d_box, draw_xyz_axis,
    )
    
    scorer = ScorePredictor()
    refiner = PoseRefinePredictor()
    glctx = dr.RasterizeCudaContext()
    est = FoundationPose(
        model_pts=mesh.vertices,
        model_normals=mesh.vertex_normals,
        mesh=mesh,
        scorer=scorer,
        refiner=refiner,
        glctx=glctx,
        debug_dir=results_dir,
    )
    logging.info("Estimator initialization done")

    tracker_2D = Cutie()
    kf = KalmanFilter6D(kf_measurement_noise_scale)

    pose_seq = np.full((total_frames, 4, 4), np.nan, dtype=np.float64)
    bbox_seq = np.tile(np.array(LOST_BBOX, dtype=np.int64), (total_frames, 1))
    kf_mean, kf_covariance = None, None
    os.makedirs(os.path.join(results_dir, "ob_in_cam"), exist_ok=True)
    n_consecutive_fail = 0

    for i in range(total_frames):
        frame_filename = f"{reader.id_strs[i]}.png"
        color = reader.get_color(i)
        depth = read_depth_m(os.path.join(video_dir, "depth", frame_filename), depth_scale)   # 米

        mask_vis_file = os.path.join(mask_visualization_path, frame_filename)
        bbox_vis_file = os.path.join(bbox_visualization_path, frame_filename)

        if i == 0:
            init_mask = read_mask(os.path.join(video_dir, "masks", frame_filename))   # 0/1 标签图
            pose = est.register(K=cam_K, rgb=color, depth=depth, ob_mask=init_mask * 255, iteration=est_refine_iter)
            if est.pose_last is None:
                raise RuntimeError(
                    f"register 失败：首帧 mask 内有效深度点不足（{video_dir}），请检查 masks/000000.png 与 depth/000000.png")
            # KF 与 est.pose_last 使用同一坐标系（居中 mesh 系）；register 的返回值是原始 mesh 系，不能混用
            kf_mean, kf_covariance = kf.initiate(mat_to_pose6(est.pose_last))
            bbox_2d = tracker_2D.initialize(
                color, init_info={"mask": init_mask},
                mask_visualization_path=mask_vis_file, bbox_visualization_path=bbox_vis_file,
            )
        else:
            kf_mean_prev, kf_cov_prev = kf_mean, kf_covariance
            # 1) 预测到本帧
            kf_mean, kf_covariance = kf.predict(kf_mean, kf_covariance)
            # 2) 2D 追踪；跟丢（哨兵）时跳过 xy 修正
            bbox_2d = tracker_2D.track(
                color, mask_visualization_path=mask_vis_file, bbox_visualization_path=bbox_vis_file,
            )
            if is_valid_bbox(bbox_2d):
                u, v = bbox_center(bbox_2d)
                tx, ty = image_point_to_xy(cam_K, u, v, depth=kf_mean[2])
                kf_mean, kf_covariance = kf.update_from_xy(kf_mean, kf_covariance, np.array([tx, ty]))
            # 3) 用修正后的预测作为精修初值；初值不合理或精修失败时本帧记为 NaN，KF 状态回滚到上一帧
            init_pose = pose6_to_mat(kf_mean[:6])
            pose = None
            if is_plausible_pose(init_pose):
                est.pose_last = torch.as_tensor(
                    init_pose, dtype=torch.float32, device=est.pose_last.device,
                ).reshape(1, 4, 4)
                try:
                    pose = est.track_one(rgb=color, depth=depth, K=cam_K, iteration=track_refine_iter)
                except torch.cuda.OutOfMemoryError:
                    raise
                except RuntimeError as e:   # 含 torch.linalg 奇异矩阵等错误
                    logging.warning(f"frame {reader.id_strs[i]}: track_one 失败，本帧记为 NaN：{e}")
                if pose is not None and not is_plausible_pose(pose):
                    logging.warning(f"frame {reader.id_strs[i]}: track_one 返回不合理位姿 (tz={np.asarray(pose).reshape(4, 4)[2, 3]:.3f} m)，本帧记为 NaN")
                    pose = None
            else:
                logging.warning(f"frame {reader.id_strs[i]}: 精修初值不合理 (tz={init_pose[2, 3]:.3f} m)，本帧记为 NaN")
            if pose is None:
                kf_mean, kf_covariance = kf_mean_prev, kf_cov_prev
                n_consecutive_fail += 1
                if n_consecutive_fail == 30:
                    logging.warning(f"已连续 {n_consecutive_fail} 帧精修失败（最近一帧 {reader.id_strs[i]}），物体可能已丢失；继续运行，这些帧记为 NaN")
            else:
                n_consecutive_fail = 0
                # 4) 用本帧精修结果修正状态
                kf_mean, kf_covariance = kf.update(kf_mean, kf_covariance, mat_to_pose6(est.pose_last))

        bbox_seq[i] = np.asarray(bbox_2d, dtype=np.int64)
        vis_color = color
        if pose is not None:
            pose = np.asarray(pose).reshape(4, 4)
            pose_seq[i] = pose
            np.savetxt(os.path.join(results_dir, "ob_in_cam", f"{reader.id_strs[i]}.txt"), pose)
            center_pose = pose @ np.linalg.inv(to_origin)
            try:
                vis_color = draw_posed_3d_box(cam_K, img=color.copy(), ob_in_cam=center_pose, bbox=bbox)
                vis_color = draw_xyz_axis(
                    vis_color, ob_in_cam=center_pose, scale=0.1, K=cam_K, thickness=3, transparency=0, is_input_rgb=True,
                )
            except cv2.error as e:
                logging.warning(f"frame {reader.id_strs[i]}: 绘制失败（物体过近/出界）：{e}")
                vis_color = color
        if show_vis:
            cv2.imshow('pose visualisation', vis_color[..., ::-1])
            cv2.waitKey(1)
        imageio.imwrite(os.path.join(pose_visualization_path, frame_filename), vis_color)

    n_nan = int(np.isnan(pose_seq).any(axis=(1, 2)).sum())
    logging.info(f"{video_dir}: {total_frames} 帧，其中 {n_nan} 帧无有效位姿")
    np.save(pose_output_path, pose_seq)
    np.savetxt(os.path.join(results_dir, "bbox_2d.txt"), bbox_seq, fmt="%d")
    if show_vis:
        cv2.destroyAllWindows()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FoundationPose with Cutie and KF")

    parser.add_argument("--testcase", type=str, required=True, help="path to the testcase relative to project root")
    
    parser.add_argument("--est_refine_iter", type=int, default=10, help="initial refine iterations")
    parser.add_argument("--track_refine_iter", type=int, default=3, help="tracking refine iterations")
    parser.add_argument("--kf_scale", type=float, default=0.05, help="kalman filter noise scale")
    parser.add_argument("--no_vis", action="store_true", help="不弹出 cv2 窗口")

    args = parser.parse_args()

    video_dir = os.path.join(src_path, "offline", args.testcase)
    results_dir = os.path.join(src_path, "results", args.testcase)

    pose_track(
        video_dir=video_dir,
        results_dir=results_dir,
        est_refine_iter=args.est_refine_iter,
        track_refine_iter=args.track_refine_iter,
        kf_measurement_noise_scale=args.kf_scale,
        show_vis=not args.no_vis,
    )

    torch.cuda.empty_cache()
