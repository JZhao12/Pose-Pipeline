"""utils.pose_utils 的单元测试（纯 CPU）。"""
import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation

from utils.pose_utils import (
    LOST_BBOX, image_point_to_xy, is_valid_bbox,
    mat_to_pose6, pose6_to_mat, project_point, to_numpy_mat,
)

# 欧拉角（xyz，弧度）边界用例：接近 ±π 绕回、接近万向锁（ry ≈ π/2）
EDGE_EULERS = [
    [3.10, 0.1, 0.5],
    [-3.13, 0.1, 0.5],
    [0.3, np.pi / 2 - 1e-3, 0.5],
    [0.3, np.pi / 2 + 1e-3, 0.5],
]


def _make_mat(rot_matrix, t):
    mat = np.eye(4)
    mat[:3, :3] = rot_matrix
    mat[:3, 3] = t
    return mat


def test_mat_pose6_roundtrip():
    rng = np.random.default_rng(0)
    rots = list(Rotation.random(50, random_state=0).as_matrix())
    rots += [Rotation.from_euler("xyz", e).as_matrix() for e in EDGE_EULERS]
    for k, rot in enumerate(rots):
        mat = _make_mat(rot, rng.uniform(-1.0, 1.0, size=3))
        back = pose6_to_mat(mat_to_pose6(mat))
        assert np.allclose(back, mat, atol=1e-9), f"往返不一致: case {k}"


def test_to_numpy_mat_accepts_torch():
    torch = pytest.importorskip("torch")
    for t in (torch.eye(4).reshape(1, 4, 4), torch.eye(4)):
        out = to_numpy_mat(t)
        assert out.shape == (4, 4)
        assert out.dtype == np.float64
        assert np.allclose(out, np.eye(4))
    # mat_to_pose6 也可以直接吃 torch 位姿
    assert np.allclose(mat_to_pose6(torch.eye(4).reshape(1, 4, 4)), np.zeros(6))


def test_is_valid_bbox():
    assert not is_valid_bbox(LOST_BBOX)
    assert not is_valid_bbox([0, 0, 0, 0])
    assert is_valid_bbox([10, 20, 30, 40])
    assert not is_valid_bbox(None)


def test_image_point_to_xy_inverts_project_point():
    K = np.array([[605.0, 0.0, 323.0], [0.0, 605.0, 244.0], [0.0, 0.0, 1.0]])
    rng = np.random.default_rng(1)
    for _ in range(20):
        rot = Rotation.random(random_state=int(rng.integers(0, 2 ** 31 - 1))).as_matrix()
        t = np.array([rng.uniform(-0.3, 0.3), rng.uniform(-0.2, 0.2), rng.uniform(0.3, 1.5)])
        pose = _make_mat(rot, t)
        # 物体系原点投影到像素，再用其深度（= tz）反投影，应回到 (tx, ty)
        u, v = project_point(K, pose, [0.0, 0.0, 0.0])
        tx, ty = image_point_to_xy(K, u, v, depth=t[2])
        assert abs(tx - t[0]) < 1e-9
        assert abs(ty - t[1]) < 1e-9


def test_is_plausible_pose():
    from utils.pose_utils import is_plausible_pose, MIN_DEPTH_M, MAX_DEPTH_M
    def with_tz(tz):
        m = np.eye(4); m[2, 3] = tz; return m
    assert is_plausible_pose(with_tz(0.6))
    assert not is_plausible_pose(with_tz(0.0))
    assert not is_plausible_pose(with_tz(-1.0))
    assert not is_plausible_pose(with_tz(MAX_DEPTH_M + 1))
    assert not is_plausible_pose(with_tz(MIN_DEPTH_M / 2))
    bad = with_tz(0.6); bad[0, 0] = np.nan
    assert not is_plausible_pose(bad)
    assert is_plausible_pose(torch.as_tensor(with_tz(0.6)).reshape(1, 4, 4))


def test_read_depth_and_mask(tmp_path):
    import cv2
    from utils.pose_utils import read_depth_m, read_mask, check_sequence_dir
    d = tmp_path / "seq_rgb_in_path"            # 路径含 rgb 子串也必须正常
    (d / "depth").mkdir(parents=True); (d / "masks").mkdir(); (d / "rgb").mkdir(); (d / "mesh").mkdir()
    raw = np.zeros((4, 5), np.uint16); raw[1, 2] = 1234
    cv2.imwrite(str(d / "depth" / "000000.png"), raw)
    depth = read_depth_m(str(d / "depth" / "000000.png"), 0.001)
    assert depth.dtype == np.float32 and depth.shape == (4, 5)
    assert depth[1, 2] == pytest.approx(1.234) and depth[0, 0] == 0.0
    m3 = np.zeros((4, 5, 3), np.uint8); m3[2, 3, 1] = 255
    cv2.imwrite(str(d / "masks" / "000000.png"), m3)
    m = read_mask(str(d / "masks" / "000000.png"))
    assert m.dtype == np.uint8 and m.sum() == 1 and m[2, 3] == 1
    with pytest.raises(FileNotFoundError):
        read_mask(str(d / "masks" / "missing.png"))
    with pytest.raises(FileNotFoundError):
        check_sequence_dir(str(d))               # 缺 rgb 图与 txt
    cv2.imwrite(str(d / "rgb" / "000000.png"), np.zeros((4, 5, 3), np.uint8))
    np.savetxt(str(d / "cam_K.txt"), np.eye(3)); np.savetxt(str(d / "depth_scale.txt"), [0.001])
    (d / "mesh" / "mesh.obj").write_text("v 0 0 0\n")
    assert check_sequence_dir(str(d)) == 1
