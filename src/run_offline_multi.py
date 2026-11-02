#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 11:25:27 2026
@author: Jianyu Zhao
"""

import argparse
import glob
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

# 路径配置
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
foundationpose_path = os.path.join(src_path, "FoundationPose")
if src_path not in sys.path:
    sys.path.append(src_path)
if foundationpose_path not in sys.path:
    sys.path.append(foundationpose_path)

from FoundationPose.estimater import (
    ScorePredictor, PoseRefinePredictor, dr,
    FoundationPose, draw_posed_3d_box, draw_xyz_axis,
)

# --- 核心推理函数 ---

def pose_track(
        testcases: list,
        est_refine_iter: int,
        track_refine_iter: int,
        kf_scale: float,
        show_vis: bool = True,
):
    if show_vis and sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
        logging.warning("未检测到 DISPLAY，自动关闭 cv2 窗口（等价于 --no_vis）")
        show_vis = False
    from FoundationPose.datareader import YcbineoatReader
    
    # 1. 初始化基础组件 (Scorer/Refiner 全局共用)
    scorer = ScorePredictor()
    refiner = PoseRefinePredictor()
    glctx = dr.RasterizeCudaContext()
    tracker_2D = Cutie()
    
    # 以第一个 testcase 为准读取相机内参和深度缩放
    base_video_dir = os.path.join(src_path, "offline", testcases[0])
    reader = YcbineoatReader(video_dir=base_video_dir, shorter_side=None, zfar=np.inf)
    cam_K = reader.K
    depth_scale = np.loadtxt(f'{base_video_dir}/depth_scale.txt')
    total_frames = len(reader)

    # 2. 为每个物体准备独立的 Estimator, KF 和结果存储
    estimators, kfs, kf_states, pose_seqs, bbox_seqs, object_meta = [], [], [], [], [], []
    
    for tc in testcases:
        tc_video_dir = os.path.join(src_path, "offline", tc)
        check_sequence_dir(tc_video_dir)
        tc_results_dir = os.path.join(src_path, "results", tc)
        os.makedirs(os.path.join(tc_results_dir, "ob_in_cam"), exist_ok=True)
        os.makedirs(os.path.join(tc_results_dir, "pose_visualization"), exist_ok=True)
        
        # 加载模型
        mesh = trimesh.load(os.path.join(tc_video_dir, "mesh", "mesh.obj"), force="mesh")
        if float(np.max(mesh.extents)) > 5.0:
            logging.warning(f"mesh 最大边长 {np.max(mesh.extents):.1f}，疑似毫米单位；流水线要求米")
        to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
        bbox = np.stack([-extents / 2, extents / 2], axis=0).reshape(2, 3)
        
        estimators.append(FoundationPose(
            model_pts=mesh.vertices, model_normals=mesh.vertex_normals, 
            mesh=mesh, scorer=scorer, refiner=refiner, glctx=glctx,
            debug_dir=tc_results_dir,
        ))
        kfs.append(KalmanFilter6D(kf_scale))
        kf_states.append([None, None]) # [mean, covariance]
        pose_seqs.append(np.full((total_frames, 4, 4), np.nan))
        bbox_seqs.append(np.tile(np.array(LOST_BBOX, dtype=np.int64), (total_frames, 1)))
        object_meta.append({'to_origin': to_origin, 'bbox': bbox, 'res_dir': tc_results_dir})

    # 多物体共享第一个 testcase 的 RGB-D 流：其余 testcase 的内参、深度比例、帧数必须一致
    for tc in testcases[1:]:
        d = os.path.join(src_path, "offline", tc)
        if not np.allclose(np.loadtxt(os.path.join(d, "cam_K.txt")).reshape(3, 3), cam_K, atol=1e-3):
            raise ValueError(f"{tc} 的 cam_K.txt 与 {testcases[0]} 不一致，多物体模式要求同一相机")
        if abs(float(np.loadtxt(os.path.join(d, "depth_scale.txt"))) - float(depth_scale)) > 1e-9:
            raise ValueError(f"{tc} 的 depth_scale.txt 与 {testcases[0]} 不一致")
        n = len(glob.glob(os.path.join(d, "rgb", "*.png")))
        if n != total_frames:
            raise ValueError(f"{tc} 有 {n} 帧，{testcases[0]} 有 {total_frames} 帧，多物体模式要求同一段录像")

    fail_counts = [0] * len(testcases)

    # 3. 逐帧追踪
    for i in range(total_frames):
        frame_id = reader.id_strs[i]
        color = reader.get_color(i)
        depth = read_depth_m(os.path.join(src_path, "offline", testcases[0], "depth", f"{frame_id}.png"), depth_scale)   # 米
        
        vis_color = color.copy()

        # --- 2D 追踪阶段 ---
        if i == 0:
            h, w = color.shape[:2]
            combined_mask = np.zeros((h, w), dtype=np.uint8)
            for idx, tc in enumerate(testcases):
                m = read_mask(os.path.join(src_path, "offline", tc, "masks", f"{frame_id}.png"))   # 缺失会直接抛 FileNotFoundError
                if m.shape != (h, w):
                    m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
                overlap = int(np.count_nonzero(combined_mask[m > 0]))
                if overlap:
                    logging.warning(f"{tc} 的首帧 mask 与前面物体重叠 {overlap} 像素，后者覆盖前者")
                combined_mask[m > 0] = idx + 1
            tracker_2D.initialize(color, init_info={"mask": combined_mask})
            all_masks = combined_mask
        else:
            # H×W uint8 标签图，像素值 = 物体编号，0 为背景
            all_masks = tracker_2D.track_mask(color)

        # --- 3D 估计与 KF 更新阶段 ---
        for idx in range(len(testcases)):
            est = estimators[idx]
            kf_obj = kfs[idx]
            meta = object_meta[idx]
            obj_id = idx + 1 # 对应 mask 中的像素值

            # 提取当前物体的 BBox；mask 为空（跟丢）时记为哨兵值
            curr_obj_mask = (all_masks == obj_id).astype(np.uint8)
            x_b, y_b, w_b, h_b = [int(v) for v in cv2.boundingRect(curr_obj_mask)]
            bbox_2d = [x_b, y_b, w_b, h_b] if (w_b > 0 and h_b > 0) else list(LOST_BBOX)
            
            if i == 0:
                # 第一帧：注册位姿
                pose = est.register(K=cam_K, rgb=color, depth=depth, ob_mask=curr_obj_mask*255, iteration=est_refine_iter)
                if est.pose_last is None:
                    raise RuntimeError(
                        f"register 失败：{testcases[idx]} 首帧 mask 内有效深度点不足，请检查其 masks/000000.png 与 depth/000000.png")
                kf_states[idx] = kf_obj.initiate(mat_to_pose6(est.pose_last))   # 与 pose_last 同坐标系
            else:
                kf_mean, kf_cov = kf_states[idx]
                kf_prev = (kf_mean, kf_cov)
                # 1. 预测到本帧
                kf_mean, kf_cov = kf_obj.predict(kf_mean, kf_cov)
                # 2. 如果 2D 追踪到了，使用中心点修正 KF
                if is_valid_bbox(bbox_2d):
                    u, v = bbox_center(bbox_2d)
                    tx, ty = image_point_to_xy(cam_K, u, v, depth=kf_mean[2])
                    kf_mean, kf_cov = kf_obj.update_from_xy(kf_mean, kf_cov, np.array([tx, ty]))
                # 3. 同步到 FoundationPose 并精修；初值不合理或精修失败时本物体本帧记为 NaN，KF 回滚
                init_pose = pose6_to_mat(kf_mean[:6])
                pose = None
                if is_plausible_pose(init_pose):
                    est.pose_last = torch.as_tensor(init_pose, dtype=torch.float32, device="cuda").reshape(1, 4, 4)
                    try:
                        pose = est.track_one(rgb=color, depth=depth, K=cam_K, iteration=track_refine_iter)
                    except torch.cuda.OutOfMemoryError:
                        raise
                    except RuntimeError as e:
                        logging.warning(f"frame {frame_id} {testcases[idx]}: track_one 失败，本帧记为 NaN：{e}")
                    if pose is not None and not is_plausible_pose(pose):
                        logging.warning(f"frame {frame_id} {testcases[idx]}: track_one 返回不合理位姿 (tz={np.asarray(pose).reshape(4, 4)[2, 3]:.3f} m)，本帧记为 NaN")
                        pose = None
                else:
                    logging.warning(f"frame {frame_id} {testcases[idx]}: 精修初值不合理 (tz={init_pose[2, 3]:.3f} m)，本帧记为 NaN")
                if pose is None:
                    kf_states[idx] = kf_prev
                    fail_counts[idx] += 1
                    if fail_counts[idx] == 30:
                        logging.warning(f"{testcases[idx]} 已连续 {fail_counts[idx]} 帧精修失败（最近一帧 {frame_id}），该物体可能已丢失；继续运行，这些帧记为 NaN")
                else:
                    fail_counts[idx] = 0
                    # 4. 用本帧精修结果修正状态
                    kf_mean, kf_cov = kf_obj.update(kf_mean, kf_cov, mat_to_pose6(est.pose_last))
                    kf_states[idx] = (kf_mean, kf_cov)

            # --- 保存与可视化 ---
            bbox_seqs[idx][i] = bbox_2d
            if pose is None:
                continue   # 本物体本帧保持 NaN，不写 txt、不画框，继续处理下一个物体
            pose_mat = np.asarray(pose).reshape(4, 4)
            pose_seqs[idx][i] = pose_mat
            np.savetxt(os.path.join(meta['res_dir'], "ob_in_cam", f"{frame_id}.txt"), pose_mat)

            # 在同一张图上绘制所有物体
            center_pose = pose_mat @ np.linalg.inv(meta['to_origin'])
            try:
                vis_color = draw_posed_3d_box(cam_K, img=vis_color, ob_in_cam=center_pose, bbox=meta['bbox'])
                vis_color = draw_xyz_axis(vis_color, ob_in_cam=center_pose, scale=0.1, K=cam_K, thickness=3, is_input_rgb=True)
            except cv2.error as e:
                logging.warning(f"frame {frame_id}: 绘制失败（物体过近/出界）：{e}")

        if show_vis:
            cv2.imshow('Multi-Object Pose Tracking', vis_color[...,::-1])
            if cv2.waitKey(1) & 0xFF == ord('q'):
                logging.warning("用户按 q 中断，剩余帧将记为 NaN")
                break
        
        # 统一保存可视化结果（可以存到第一个物体的目录或新建）
        imageio.imwrite(os.path.join(object_meta[0]['res_dir'], "pose_visualization", f"{frame_id}.png"), vis_color)

    # 循环结束后保存所有 npy 与 2D bbox
    for idx, tc in enumerate(testcases):
        res_dir = object_meta[idx]['res_dir']
        n_nan = int(np.isnan(pose_seqs[idx]).any(axis=(1, 2)).sum())
        logging.info(f"{tc}: {total_frames} 帧，其中 {n_nan} 帧无有效位姿")
        np.save(os.path.join(res_dir, "pose.npy"), pose_seqs[idx])
        np.savetxt(os.path.join(res_dir, "bbox_2d.txt"), bbox_seqs[idx], fmt="%d")

    if show_vis:
        cv2.destroyAllWindows()
    torch.cuda.empty_cache()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 支持传入多个 testcase，用逗号分隔，例如 --testcases "teabox,medbox"
    parser.add_argument("--testcases", type=str, required=True, help="comma separated testcase names")
    parser.add_argument("--est_refine_iter", type=int, default=10)
    parser.add_argument("--track_refine_iter", type=int, default=3)
    parser.add_argument("--kf_scale", type=float, default=0.05)
    parser.add_argument("--no_vis", action="store_true", help="不弹出 cv2 窗口")
    args = parser.parse_args()

    tc_list = [name.strip() for name in args.testcases.split(',')]
    pose_track(tc_list, args.est_refine_iter, args.track_refine_iter, args.kf_scale, show_vis=not args.no_vis)
