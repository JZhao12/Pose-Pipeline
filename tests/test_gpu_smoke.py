"""GPU 冒烟测试：在 tmp 沙盒里用软链接复用源码/权重/数据，只跑前几帧，结果写到沙盒的 results/，不污染仓库。

默认跳过；设置 PP_GPU_TESTS=1 才运行。
"""
import os
import sys
import shutil
import subprocess
import pytest
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
pytestmark = pytest.mark.skipif(os.environ.get("PP_GPU_TESTS") != "1", reason="设置 PP_GPU_TESTS=1 才运行 GPU 冒烟测试")

TESTCASES = ["teabox_D435", "medbox_D435"]
NUM_FRAMES = 6


@pytest.fixture
def sandbox(tmp_path):
    """迷你项目根：src/FoundationPose/Cutie 软链到真实目录，offline 只含前 NUM_FRAMES 帧。

    脚本用 os.path.abspath(__file__/..) 推项目根，abspath 不解析软链接，
    所以 src_path 就是这个沙盒目录，results/ 会写到沙盒里。
    """
    for name in ("src", "FoundationPose", "Cutie"):
        os.symlink(os.path.join(ROOT, name), str(tmp_path / name))
    (tmp_path / "results").mkdir()

    for tc in TESTCASES:
        src_dir = os.path.join(ROOT, "offline", tc)
        if not os.path.isdir(src_dir):
            pytest.skip(f"缺少数据集 {src_dir}")
        dst_dir = tmp_path / "offline" / tc
        for sub in ("rgb", "depth"):
            (dst_dir / sub).mkdir(parents=True)
            for fn in sorted(os.listdir(os.path.join(src_dir, sub)))[:NUM_FRAMES]:
                os.symlink(os.path.join(src_dir, sub, fn), str(dst_dir / sub / fn))
        (dst_dir / "masks").mkdir()
        os.symlink(os.path.join(src_dir, "masks", "000000.png"), str(dst_dir / "masks" / "000000.png"))
        os.symlink(os.path.join(src_dir, "mesh"), str(dst_dir / "mesh"))
        for fn in ("cam_K.txt", "depth_scale.txt"):
            shutil.copy(os.path.join(src_dir, fn), str(dst_dir / fn))
    return tmp_path


def _run(sandbox, cmd):
    proc = subprocess.run(
        [sys.executable] + cmd,
        cwd=str(sandbox),
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        capture_output=True,
        text=True,
        timeout=900,
    )
    assert proc.returncode == 0, (
        f"returncode={proc.returncode}\n--- stdout (tail) ---\n{proc.stdout[-2000:]}"
        f"\n--- stderr (tail) ---\n{proc.stderr[-4000:]}"
    )


def _count(path, suffix):
    return len([f for f in os.listdir(path) if f.endswith(suffix)])


def _check_outputs(res_dir):
    poses = np.load(res_dir / "pose.npy")
    assert poses.shape == (NUM_FRAMES, 4, 4)
    assert not np.isnan(poses).any()
    bboxes = np.loadtxt(res_dir / "bbox_2d.txt", dtype=np.int64).reshape(-1, 4)
    assert bboxes.shape[0] == NUM_FRAMES


def test_single_object_runs(sandbox):
    _run(sandbox, [
        "src/run_offline.py", "--testcase", "teabox_D435", "--no_vis",
        "--est_refine_iter", "3", "--track_refine_iter", "2",
    ])
    res = sandbox / "results" / "teabox_D435"
    _check_outputs(res)
    assert _count(res / "ob_in_cam", ".txt") == NUM_FRAMES
    for sub in ("mask_visualization", "bbox_visualization", "pose_visualization"):
        assert _count(res / sub, ".png") == NUM_FRAMES, sub


def test_multi_object_runs(sandbox):
    _run(sandbox, [
        "src/run_offline_multi.py", "--testcases", "teabox_D435,medbox_D435", "--no_vis",
        "--est_refine_iter", "3", "--track_refine_iter", "2",
    ])
    for tc in TESTCASES:
        _check_outputs(sandbox / "results" / tc)
    assert _count(sandbox / "results" / "teabox_D435" / "pose_visualization", ".png") == NUM_FRAMES
