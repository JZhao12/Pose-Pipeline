"""KalmanFilter6D 的单元测试（纯 CPU）。"""
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from utils.kalman_filter_6d import KalmanFilter6D
from utils.pose_utils import pose6_to_mat

KF_SCALE = 0.05  # 与 CLI 默认的 --kf_scale 一致


def pose6(t, euler):
    """平移 + 欧拉角（xyz，弧度）→ [t, rotvec]。"""
    return np.r_[np.asarray(t, dtype=np.float64), Rotation.from_euler("xyz", euler).as_rotvec()]


def rot_err_deg(rotvec_a, rotvec_b):
    """两个旋转向量对应旋转之间的测地角（度）。"""
    ra = Rotation.from_rotvec(rotvec_a)
    rb = Rotation.from_rotvec(rotvec_b)
    return float(np.rad2deg((ra * rb.inv()).magnitude()))


@pytest.mark.parametrize("euler", [
    [0.0, 0.0, 0.0],
    [0.3, 0.2, 0.0],
    [0.3, -0.2, 1.0],
    [3.1, 0.0, -3.1],
])
def test_static_any_orientation_no_linalg_error(euler):
    kf = KalmanFilter6D(KF_SCALE)
    meas = pose6([0.05, -0.03, 0.6], euler)
    mean, cov = kf.initiate(meas)
    for _ in range(200):
        mean, cov = kf.predict(mean, cov)
        mean, cov = kf.update(mean, cov, meas)
        assert np.all(np.isfinite(mean))
        assert np.all(np.isfinite(cov))
        assert np.allclose(cov, cov.T, rtol=1e-9, atol=1e-15)
    assert np.linalg.norm(mean[:3] - meas[:3]) < 1e-4      # < 0.1 mm
    assert rot_err_deg(mean[3:6], meas[3:6]) < 0.01        # < 0.01°


def test_rotation_crossing_pi_stays_continuous():
    kf = KalmanFilter6D(KF_SCALE)
    t = [0.05, -0.03, 0.6]
    # rx 从 +3.10 越过 ±π 到 -3.07，真实旋转是连续的小步长
    truths = [pose6(t, [rx, 0.1, 0.5]) for rx in (3.10, 3.13, -3.13, -3.10, -3.07)]
    mean, cov = kf.initiate(truths[0])
    for k, truth in enumerate(truths[1:], start=1):
        mean, cov = kf.predict(mean, cov)
        mean, cov = kf.update(mean, cov, truth)
        # 用旋转矩阵计算相对真值的测地角
        r_est = pose6_to_mat(mean[:6])[:3, :3]
        r_true = pose6_to_mat(truth)[:3, :3]
        err = np.rad2deg((Rotation.from_matrix(r_est) * Rotation.from_matrix(r_true).inv()).magnitude())
        assert err < 1.5, f"step {k}: 旋转误差 {err:.3f} deg"
        assert np.linalg.norm(mean[9:12]) < np.deg2rad(10.0), f"step {k}: 角速度估计异常"


def test_constant_velocity_prediction_converges():
    """匀速模型本身：显式使用 CV 旋转模型（默认是旋转随机游走），旋转过程噪声沿用原来的 0.5°。"""
    kf = KalmanFilter6D(KF_SCALE, vel_alpha_rot=1.0, proc_std_rot=np.deg2rad(0.5))
    step_t = np.array([0.005, 0.0, 0.0])                       # 每帧平移 5 mm
    step_rot = Rotation.from_euler("z", 1.0, degrees=True)     # 每帧绕 z 轴 +1°
    t = np.array([0.0, 0.0, 0.6])
    rot = Rotation.identity()
    mean, cov = kf.initiate(np.r_[t, rot.as_rotvec()])
    for k in range(1, 41):
        t = t + step_t
        rot = step_rot * rot                                   # 旋转合成生成真值
        truth = np.r_[t, rot.as_rotvec()]
        mean, cov = kf.predict(mean, cov)
        # 先比较"预测均值 vs 真值"，再用真值做更新
        t_err = np.linalg.norm(mean[:3] - truth[:3])
        r_err = rot_err_deg(mean[3:6], truth[3:6])
        if k >= 15:
            assert t_err < 5e-4, f"frame {k}: 平移预测误差 {t_err * 1e3:.4f} mm"
            assert r_err < 0.1, f"frame {k}: 旋转预测误差 {r_err:.4f} deg"
        mean, cov = kf.update(mean, cov, truth)


def test_xy_noise_independent_of_lateral_offset():
    kf = KalmanFilter6D(KF_SCALE)
    rotvec = Rotation.from_euler("xyz", [0.1, 0.2, 0.3]).as_rotvec()
    _, cov = kf.initiate(np.r_[[0.0, 0.0, 0.6], rotvec])
    mean_a = np.r_[[0.0, 0.0, 0.6], rotvec, np.zeros(6)]
    mean_b = np.r_[[0.2, 0.15, 0.6], rotvec, np.zeros(6)]

    # 同一深度下，xy 观测噪声与横向偏移无关
    _, proj_cov_a = kf.project_for_xy(mean_a, cov)
    _, proj_cov_b = kf.project_for_xy(mean_b, cov)
    assert np.allclose(proj_cov_a, proj_cov_b)

    # 对 20 mm 偏差的响应比例应在 (0, 1) 之间
    offset = 0.02
    for mean in (mean_a, mean_b):
        new_mean, _ = kf.update_from_xy(mean, cov, mean[:2] + np.array([offset, 0.0]))
        ratio = (new_mean[0] - mean[0]) / offset
        assert 0.0 < ratio < 1.0


def test_initiate_shapes():
    kf = KalmanFilter6D(KF_SCALE, init_std_t=0.02, init_std_rot=0.03, init_std_vt=0.04, init_std_w=0.05)
    meas = pose6([0.1, 0.2, 0.7], [0.1, 0.2, 0.3])
    mean, cov = kf.initiate(meas)
    assert mean.shape == (12,)
    assert cov.shape == (12, 12)
    assert np.allclose(mean[:6], meas)
    assert np.allclose(mean[6:], 0.0)
    expected = np.r_[[0.02] * 3, [0.03] * 3, [0.04] * 3, [0.05] * 3] ** 2
    assert np.allclose(np.diag(cov), expected)
    assert np.allclose(cov, np.diag(np.diag(cov)))


def test_update_moves_toward_measurement():
    kf = KalmanFilter6D(KF_SCALE)
    base = pose6([0.0, 0.0, 0.6], [0.1, 0.2, 0.3])
    mean, cov = kf.initiate(base)

    # 观测相对状态偏 10 mm（x 方向）和 5°（绕相机 z 轴）
    off_t = np.array([0.01, 0.0, 0.0])
    off_rot = Rotation.from_euler("z", 5.0, degrees=True)
    meas = np.r_[base[:3] + off_t, (off_rot * Rotation.from_rotvec(base[3:6])).as_rotvec()]

    new_mean, _ = kf.update(mean, cov, meas)

    # 平移：误差减小且不越过观测，增益在 (0, 1)
    gain_t = (new_mean[0] - base[0]) / off_t[0]
    assert 0.0 < gain_t < 1.0
    assert np.linalg.norm(new_mean[:3] - meas[:3]) < np.linalg.norm(base[:3] - meas[:3])

    # 旋转：沿测地线朝观测移动，既没动 0 也没越过
    moved = rot_err_deg(new_mean[3:6], base[3:6])
    remain = rot_err_deg(new_mean[3:6], meas[3:6])
    assert 0.0 < moved / 5.0 < 1.0
    assert remain < 5.0
    assert moved + remain == pytest.approx(5.0, abs=1e-6)


@pytest.mark.parametrize("alpha_rot", [0.0, 0.5])
def test_vel_alpha_rot_limits_rotation_extrapolation(alpha_rot):
    """角速度保留系数：α=0 时旋转不外推；α=0.5 时外推量减半。"""
    kf = KalmanFilter6D(KF_SCALE, vel_alpha_rot=alpha_rot)
    base = pose6([0.0, 0.0, 0.6], [0.1, 0.2, 0.3])
    mean, cov = kf.initiate(base)
    mean[9:12] = np.deg2rad(np.array([0.0, 0.0, 4.0]))      # 人为给 4°/帧 的角速度
    mean_pred, cov_pred = kf.predict(mean, cov)
    moved = rot_err_deg(mean_pred[3:6], base[3:6])
    assert moved == pytest.approx(4.0 * alpha_rot, abs=1e-6)
    assert np.allclose(mean_pred[9:12], alpha_rot * mean[9:12])
    assert np.all(np.isfinite(cov_pred)) and np.allclose(cov_pred, cov_pred.T)


def test_vel_alpha_t_zero_is_random_walk():
    kf = KalmanFilter6D(KF_SCALE, vel_alpha_t=0.0)
    base = pose6([0.0, 0.0, 0.6], [0.0, 0.0, 0.0])
    mean, cov = kf.initiate(base)
    mean[6:9] = [0.01, 0.0, 0.0]
    mean_pred, _ = kf.predict(mean, cov)
    assert np.allclose(mean_pred[:3], base[:3])
    assert np.allclose(mean_pred[6:9], 0.0)


def test_default_rotation_is_random_walk():
    """默认 vel_alpha_rot=0：predict 不改变旋转，角速度状态恒为 0；update 后旋转近似直通观测。"""
    kf = KalmanFilter6D(KF_SCALE)
    assert kf.vel_alpha_rot == 0.0 and kf.vel_alpha_t == 1.0
    base = pose6([0.0, 0.0, 0.6], [0.1, 0.2, 0.3])
    mean, cov = kf.initiate(base)
    step = Rotation.from_euler("z", 3.0, degrees=True)
    rot = Rotation.from_rotvec(base[3:6])
    for _ in range(10):
        rot = step * rot
        truth = np.r_[base[:3], rot.as_rotvec()]
        mean_pred, cov = kf.predict(mean, cov)
        assert rot_err_deg(mean_pred[3:6], mean[3:6]) < 1e-9        # 旋转不外推
        assert np.allclose(mean_pred[9:12], 0.0)                     # 角速度恒为 0
        mean, cov = kf.update(mean_pred, cov, truth)
    assert rot_err_deg(mean[3:6], truth[3:6]) < 0.3                  # 3°/帧步进下 update 后残差很小
