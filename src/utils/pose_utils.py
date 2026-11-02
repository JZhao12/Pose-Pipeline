"""位姿与几何辅助函数。

约定：
- 4x4 位姿矩阵 ob_in_cam，平移单位米。
- 6 维位姿 pose6 = [tx, ty, tz, r1, r2, r3]，后三维是旋转向量（轴角，弧度）。
  它只是旋转矩阵的紧凑存储，不要对其做线性加减（KalmanFilter6D 内部用旋转合成处理）。
- 2D bbox 为 [x, y, w, h] 像素；跟丢哨兵为 LOST_BBOX。
"""
import glob
import os

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

LOST_BBOX = [-1, -1, 0, 0]


def to_numpy_mat(pose):
    """torch (1,4,4)/(4,4) 或 numpy → (4,4) float64。"""
    try:
        import torch
        if torch.is_tensor(pose):
            pose = pose.detach().cpu().numpy()
    except ImportError:
        pass
    return np.asarray(pose, dtype=np.float64).reshape(4, 4)


def mat_to_pose6(pose):
    mat = to_numpy_mat(pose)
    return np.r_[mat[:3, 3], Rotation.from_matrix(mat[:3, :3]).as_rotvec()]


def pose6_to_mat(pose6):
    pose6 = np.asarray(pose6, dtype=np.float64).reshape(6)
    mat = np.eye(4)
    mat[:3, :3] = Rotation.from_rotvec(pose6[3:]).as_matrix()
    mat[:3, 3] = pose6[:3]
    return mat


def is_valid_bbox(bbox):
    if bbox is None or len(bbox) != 4:
        return False
    x, y, w, h = bbox
    return x >= 0 and y >= 0 and w > 0 and h > 0


def bbox_center(bbox):
    x, y, w, h = bbox
    return x + w / 2.0, y + h / 2.0


def image_point_to_xy(K, u, v, depth):
    """像素点 (u, v) 在深度 depth（米）处反投影得到相机系 (tx, ty)。"""
    K = np.asarray(K, dtype=np.float64)
    return float((u - K[0, 2]) * depth / K[0, 0]), float((v - K[1, 2]) * depth / K[1, 1])


def project_point(K, ob_in_cam, point_3d):
    """物体系 3D 点 → 像素 (u, v)，float。"""
    p = to_numpy_mat(ob_in_cam) @ np.r_[np.asarray(point_3d, dtype=np.float64).reshape(3), 1.0]
    K = np.asarray(K, dtype=np.float64)
    return float(K[0, 0] * p[0] / p[2] + K[0, 2]), float(K[1, 1] * p[1] / p[2] + K[1, 2])


MIN_DEPTH_M = 0.05   # 物体中心不可能比这更靠近相机
MAX_DEPTH_M = 10.0   # 深度相机有效量程上限


def is_plausible_pose(pose):
    """位姿是否物理合理：元素全部有限，且物体中心深度 tz 在 (MIN_DEPTH_M, MAX_DEPTH_M) 内。"""
    mat = to_numpy_mat(pose)
    return bool(np.all(np.isfinite(mat)) and MIN_DEPTH_M < mat[2, 3] < MAX_DEPTH_M)


def read_depth_m(path, depth_scale):
    """读 16 位深度 PNG 并换算成米；无效像素（raw==0）保持 0。"""
    raw = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise FileNotFoundError(f"深度图读取失败：{path}")
    return raw.astype(np.float32) * float(depth_scale)


def read_mask(path):
    """读首帧 mask（灰度 0/255、0/1 或三通道都接受），返回 H×W uint8 的 0/1 图。"""
    m = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if m is None:
        raise FileNotFoundError(f"mask 读取失败：{path}")
    if m.ndim == 3:
        m = m.max(axis=2)
    return (m > 0).astype(np.uint8)


def check_sequence_dir(video_dir):
    """运行前检查 testcase 目录结构，缺什么就报什么，返回 (rgb 文件数)。"""
    if not os.path.isdir(video_dir):
        raise FileNotFoundError(f"testcase 目录不存在：{video_dir}")
    n_rgb = len(glob.glob(os.path.join(video_dir, "rgb", "*.png")))
    if n_rgb == 0:
        raise FileNotFoundError(f"{video_dir}/rgb/ 下没有 png 图像")
    for name in ("cam_K.txt", "depth_scale.txt", os.path.join("masks", "000000.png"), os.path.join("mesh", "mesh.obj")):
        if not os.path.exists(os.path.join(video_dir, name)):
            raise FileNotFoundError(f"缺少 {os.path.join(video_dir, name)}")
    return n_rgb
