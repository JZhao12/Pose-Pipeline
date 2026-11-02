"""6DoF 位姿卡尔曼滤波（匀速模型，dt = 1 帧）。

状态 mean（12 维）: [tx, ty, tz, r1, r2, r3, v_tx, v_ty, v_tz, w1, w2, w3]
  - t   : 物体在相机系中的平移（米）
  - r   : 名义旋转 R 的旋转向量 Log(R)（弧度）。只是 R 的存储形式，所有更新都通过旋转合成完成，
          因此不存在欧拉角 ±π 绕回和万向锁问题。
  - v_t : 平移速度（米/帧）
  - w   : 角速度（弧度/帧），左乘约定 R_{k+1} = Exp(w) R_k（相机系下的旋转增量）
covariance（12x12）定义在误差状态 [δt, δθ, δv_t, δw] 上，其中 δθ 为左扰动：R_true = Exp(δθ) R_nominal。

噪声参数全部是物理单位的常数：
  - 过程噪声：proc_std_*（每帧）
  - 6D 位姿观测噪声：meas_std_t / meas_std_rot 乘以 measurement_noise_scale（即 CLI 的 --kf_scale）。
    默认 0.06 m / 10°，kf_scale=0.05 时相当于 3 mm / 0.5°。
  - 2D 中心反投影得到的 xy 观测：std = xy_std_per_depth * depth，像素级误差随深度线性放大。

速度保留系数 vel_alpha_t / vel_alpha_rot ∈ [0, 1]：
  - 1.0：匀速外推（CV 模型）；0.0：不外推，速度状态始终为 0，等价于随机游走模型。
  - 默认平移匀速外推（vel_alpha_t=1.0）、旋转不外推（vel_alpha_rot=0.0，过程噪声 proc_std_rot=3°/帧，
    旋转近似直通精修结果）。这是闭环实测的结论：对近似旋转对称的物体（如 ball），绕对称轴的旋转
    不可观测，精修器无法纠正该方向的初值，匀速外推角速度会让它沿该轴随机游走，旋转抖动约涨 3 倍；
    而箱体类物体在两种设置下指标持平。需要旋转外推时显式传 vel_alpha_rot=1.0、proc_std_rot≈0.5°。
"""
import numpy as np
import scipy.linalg
from scipy.spatial.transform import Rotation


class KalmanFilter6D:

    def __init__(
            self,
            measurement_noise_scale: float = 1.0,
            *,
            vel_alpha_t: float = 1.0,
            vel_alpha_rot: float = 0.0,
            proc_std_t: float = 0.003,
            proc_std_rot: float = np.deg2rad(3.0),
            proc_std_vt: float = 0.005,
            proc_std_w: float = np.deg2rad(1.0),
            meas_std_t: float = 0.06,
            meas_std_rot: float = np.deg2rad(10.0),
            xy_std_per_depth: float = 0.02,
            init_std_t: float = 0.01,
            init_std_rot: float = np.deg2rad(2.0),
            init_std_vt: float = 0.01,
            init_std_w: float = np.deg2rad(2.0),
    ):
        ndim, dt = 6, 1.0
        # 状态转移 F（12x12）：位姿 += α·速度；速度 ← α·速度（α=1 为匀速模型，α=0 为随机游走）
        alphas = [vel_alpha_t] * 3 + [vel_alpha_rot] * 3
        self._motion_mat = np.eye(2 * ndim)
        for i, a in enumerate(alphas):
            self._motion_mat[i, ndim + i] = a * dt
            self._motion_mat[ndim + i, ndim + i] = a
        # 观测矩阵：6D 位姿观测 H（6x12）、xy 观测 H_xy（2x12）
        self._update_mat = np.eye(ndim, 2 * ndim)
        self._update_mat_xy = np.zeros((2, 2 * ndim))
        self._update_mat_xy[0, 0] = 1.0
        self._update_mat_xy[1, 1] = 1.0

        self.measurement_noise_scale = float(measurement_noise_scale)
        self.vel_alpha_t = vel_alpha_t
        self.vel_alpha_rot = vel_alpha_rot
        self.proc_std_t = proc_std_t
        self.proc_std_rot = proc_std_rot
        self.proc_std_vt = proc_std_vt
        self.proc_std_w = proc_std_w
        self.meas_std_t = meas_std_t
        self.meas_std_rot = meas_std_rot
        self.xy_std_per_depth = xy_std_per_depth
        self.init_std_t = init_std_t
        self.init_std_rot = init_std_rot
        self.init_std_vt = init_std_vt
        self.init_std_w = init_std_w

        assert self.measurement_noise_scale > 0, "measurement_noise_scale 必须 > 0"
        assert 0.0 <= vel_alpha_t <= 1.0 and 0.0 <= vel_alpha_rot <= 1.0, "vel_alpha_* 必须在 [0, 1] 内"
        for name in ("proc_std_t", "proc_std_rot", "proc_std_vt", "proc_std_w", "meas_std_t", "meas_std_rot",
                     "xy_std_per_depth", "init_std_t", "init_std_rot", "init_std_vt", "init_std_w"):
            assert getattr(self, name) > 0, f"{name} 必须 > 0"

    # ---------- 旋转工具 ----------
    @staticmethod
    def _rot(rotvec):
        return Rotation.from_rotvec(np.asarray(rotvec, dtype=np.float64))

    @staticmethod
    def _depth_scale(mean):
        return max(abs(float(mean[2])), 1e-2)

    def residual(self, mean, measurement):
        """观测 ⊟ 名义状态：[t_m - t, Log(R_m R^{-1})]，旋转部分为切空间小量。"""
        m = np.asarray(measurement, dtype=np.float64).reshape(6)
        d_t = m[:3] - mean[:3]
        d_theta = (self._rot(m[3:6]) * self._rot(mean[3:6]).inv()).as_rotvec()
        return np.r_[d_t, d_theta]

    def _apply_correction(self, mean, delta):
        """名义状态 ⊞ 误差状态：平移/速度线性相加，旋转左乘 Exp(δθ)。"""
        new_mean = np.array(mean, dtype=np.float64, copy=True)
        new_mean[0:3] += delta[0:3]
        new_mean[3:6] = (self._rot(delta[3:6]) * self._rot(mean[3:6])).as_rotvec()
        new_mean[6:12] += delta[6:12]
        return new_mean

    def _gain(self, covariance, update_mat, projected_cov):
        """K = P Hᵀ S⁻¹，用 Cholesky 求解。"""
        chol_factor, lower = scipy.linalg.cho_factor(projected_cov, lower=True, check_finite=False)
        return scipy.linalg.cho_solve(
            (chol_factor, lower), np.dot(covariance, update_mat.T).T, check_finite=False).T

    @staticmethod
    def _symmetrize(cov):
        return 0.5 * (cov + cov.T)

    # ---------- 滤波步骤 ----------
    def initiate(self, measurement):
        """用首帧 6D 位姿 [t, rotvec] 初始化，速度置零。"""
        measurement = np.asarray(measurement, dtype=np.float64).reshape(6)
        mean = np.r_[measurement, np.zeros(6)]
        std = ([self.init_std_t] * 3 + [self.init_std_rot] * 3 +
               [self.init_std_vt] * 3 + [self.init_std_w] * 3)
        return mean, np.diag(np.square(std))

    def predict(self, mean, covariance):
        """预测：t += α_t·v_t；R ← Exp(α_r·w) R；速度按 α 衰减。"""
        mean = np.asarray(mean, dtype=np.float64)
        a_t, a_r = self.vel_alpha_t, self.vel_alpha_rot
        new_mean = mean.copy()
        new_mean[0:3] = mean[0:3] + a_t * mean[6:9]
        new_mean[3:6] = (self._rot(a_r * mean[9:12]) * self._rot(mean[3:6])).as_rotvec()
        new_mean[6:9] = a_t * mean[6:9]
        new_mean[9:12] = a_r * mean[9:12]

        std = ([self.proc_std_t] * 3 + [self.proc_std_rot] * 3 +
               [self.proc_std_vt] * 3 + [self.proc_std_w] * 3)
        motion_cov = np.diag(np.square(std))
        new_cov = np.linalg.multi_dot((self._motion_mat, covariance, self._motion_mat.T)) + motion_cov
        return new_mean, self._symmetrize(new_cov)

    def project(self, mean, covariance):
        """6D 观测空间：返回名义位姿与 S = H P Hᵀ + R。"""
        s = self.measurement_noise_scale
        std = [s * self.meas_std_t] * 3 + [s * self.meas_std_rot] * 3
        innovation_cov = np.diag(np.square(std))
        projected_cov = np.linalg.multi_dot((self._update_mat, covariance, self._update_mat.T)) + innovation_cov
        return np.array(mean[:6], dtype=np.float64, copy=True), projected_cov

    def project_for_xy(self, mean, covariance):
        """xy 观测空间：噪声随深度线性放大，与横向偏移无关。"""
        std_xy = [self.xy_std_per_depth * self._depth_scale(mean)] * 2
        innovation_cov = np.diag(np.square(std_xy))
        projected_cov = np.linalg.multi_dot((self._update_mat_xy, covariance, self._update_mat_xy.T)) + innovation_cov
        return np.array(mean[:2], dtype=np.float64, copy=True), projected_cov

    def update(self, mean, covariance, measurement):
        """用完整 6D 位姿观测 [t, rotvec] 修正。"""
        _, projected_cov = self.project(mean, covariance)
        kalman_gain = self._gain(covariance, self._update_mat, projected_cov)
        innovation = self.residual(mean, measurement)
        new_mean = self._apply_correction(mean, np.dot(kalman_gain, innovation))
        new_cov = covariance - np.linalg.multi_dot((kalman_gain, projected_cov, kalman_gain.T))
        return new_mean, self._symmetrize(new_cov)

    def update_from_xy(self, mean, covariance, measurement_xy):
        """只用 (tx, ty) 修正。"""
        projected_mean, projected_cov = self.project_for_xy(mean, covariance)
        kalman_gain = self._gain(covariance, self._update_mat_xy, projected_cov)
        innovation = np.asarray(measurement_xy, dtype=np.float64).reshape(2) - projected_mean
        new_mean = self._apply_correction(mean, np.dot(kalman_gain, innovation))
        new_cov = covariance - np.linalg.multi_dot((kalman_gain, projected_cov, kalman_gain.T))
        return new_mean, self._symmetrize(new_cov)
