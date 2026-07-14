# -*- coding: utf-8 -*-
"""用 alicia_teleop_fr3.py 的实际滤波实现，数值验证对 8-12Hz 震颤的幅值抑制率。
场景: 125Hz 采样，1° 幅值震颤，分别叠加在静止基线和 30°/s 匀速运动上。"""
import numpy as np

# ---- 复制自 alicia_teleop_fr3.py 的两级滤波实现（参数取当前配置）----
class OneEuroFilter:
    def __init__(self, n_joints=1, mincutoff=3.0, beta=0.07, dcutoff=1.0):
        self.mincutoff, self.beta, self.dcutoff = mincutoff, beta, dcutoff
        self._x = None
        self._dx = np.zeros(n_joints)
        self._last_t = None
    def step(self, z, t):
        z = np.asarray(z, dtype=float)
        if self._x is None:
            self._x = z.copy(); self._last_t = t
            return self._x.copy()
        dt = max(1e-6, t - self._last_t); self._last_t = t
        raw_dx = (z - self._x) / dt
        alpha_d = self._alpha(dt, self.dcutoff)
        self._dx = alpha_d * raw_dx + (1.0 - alpha_d) * self._dx
        cutoff = self.mincutoff + self.beta * np.abs(self._dx)
        alpha = self._alpha(dt, cutoff)
        self._x = alpha * z + (1.0 - alpha) * self._x
        return self._x.copy()
    @staticmethod
    def _alpha(dt, cutoff):
        tau = 1.0 / (2.0 * np.pi * np.asarray(cutoff, dtype=float))
        return dt / (dt + tau)

class LowPassFilter:
    def __init__(self, n_joints=1, cutoff_hz=3.0):
        self.cutoff_hz = cutoff_hz
        self._y = None; self._last_t = None
    def step(self, x, t):
        x = np.asarray(x, dtype=float)
        if self._y is None:
            self._y = x.copy(); self._last_t = t
            return self._y.copy()
        dt = max(1e-6, t - self._last_t); self._last_t = t
        tau = 1.0 / (2.0 * np.pi * self.cutoff_hz)
        alpha = dt / (dt + tau)
        self._y = alpha * x + (1.0 - alpha) * self._y
        return self._y.copy()

RATE = 125.0
DUR = 8.0
t = np.arange(0, DUR, 1.0 / RATE)

def band_amp(sig, f_lo=7.0, f_hi=13.0):
    """稳态段 8-12Hz 频段幅值（FFT 带内峰值）"""
    s = sig[len(sig)//2:]  # 后半段稳态
    # 去线性趋势（消除匀速运动斜坡的频谱泄漏），加窗
    x = np.arange(len(s))
    s = s - np.polyval(np.polyfit(x, s, 1), x)
    s = s * np.hanning(len(s)) * 2  # hann 窗幅值补偿
    n = len(s)
    freqs = np.fft.rfftfreq(n, 1.0/RATE)
    mag = np.abs(np.fft.rfft(s)) / n * 2
    mask = (freqs >= f_lo) & (freqs <= f_hi)
    return mag[mask].max()

print(f"{'场景':<14}{'震颤Hz':>7}{'输入幅值°':>10}{'输出幅值°':>10}{'抑制率':>8}")
for freq in (8.0, 10.0, 12.0):
    for label, base in ((u'静止', np.zeros_like(t)), (u'运动30°/s', 30.0 * t)):
        tremor = 1.0 * np.sin(2 * np.pi * freq * t)
        raw = base + tremor
        f1 = OneEuroFilter(1, mincutoff=2.0, beta=0.05)   # 当前配置
        f2 = LowPassFilter(1, cutoff_hz=5.0)              # 当前配置
        out = np.array([f2.step(f1.step(np.array([v]), ti), ti)[0]
                        for v, ti in zip(raw, t)])
        a_in = band_amp(raw)
        a_out = band_amp(out)
        供 = (1 - a_out / a_in) * 100
        print(f"{label:<14}{freq:>7.0f}{a_in:>10.3f}{a_out:>10.3f}{供:>7.1f}%")
