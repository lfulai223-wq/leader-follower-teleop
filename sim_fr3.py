#!/usr/bin/env python3
"""
FR3 仿真模块 —— MockFR3 + 实时可视化

MockFR3 是 fairino.Robot.RPC() 的 drop-in 替换：
  - 实现相同的方法接口（ServoJ、GetActualJointPosDegree 等）
  - 在内存中维护仿真关节状态
  - 不需要任何网络连接或法奥 SDK

start_visualization(mock) 在主线程中启动 matplotlib 实时图表：
  - 显示 6 个关节的当前角度（柱状图）与目标角度（圆点）
  - 显示关节软限位边界
  - 20Hz 刷新，非阻塞式与遥操作主循环并行运行

使用方式（由 alicia_teleop_fr3.py --sim 调用，无需直接运行此文件）
"""

import threading
import time
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation


# FR3 各关节软限位（度）
JOINT_MIN = np.array([-175.0, -265.0, -150.0, -265.0, -175.0, -360.0])
JOINT_MAX = np.array([ 175.0,   85.0,  150.0,   85.0,  175.0,  360.0])
JOINT_NAMES = ["J1", "J2", "J3", "J4", "J5", "J6"]


class MockFR3:
    """Drop-in 替代 fairino.Robot.RPC()。

    所有方法签名与真实 SDK 完全一致，可直接替换使用。
    ServoJ 接收的关节角度立即写入内部状态（无动力学模拟，瞬时响应）。
    """

    def __init__(self, ip: str = "127.0.0.1"):
        self._ip = ip
        self._lock = threading.Lock()
        self._joints = np.zeros(6, dtype=float)   # 当前仿真关节角度（度）
        self._target = np.zeros(6, dtype=float)   # 最近一次 ServoJ 目标
        self._cmd_count = 0
        self._last_cmd_time: float = 0.0
        self._queue_len = 0
        print(f"[SIM] MockFR3 已初始化，仿真 FR3 @ {ip}")

    # ------------------------------------------------------------------ #
    #  与真实 SDK 接口一致的方法
    # ------------------------------------------------------------------ #

    def GetSDKVersion(self):
        return 0, "MockFR3-Sim-1.0 (仿真模式)"

    def GetControllerIP(self):
        return 0, self._ip

    def GetActualJointPosDegree(self, flag=1):
        with self._lock:
            return 0, self._joints.copy().tolist()

    def GetMotionQueueLength(self):
        with self._lock:
            return 0, self._queue_len

    def ServoJ(self, joint_pos, axisPos=None, acc=0.0, vel=0.0,
               cmdT=0.008, filterT=0.0, gain=0.0, id=0):
        joints = np.array(joint_pos[:6], dtype=float)
        with self._lock:
            self._target = joints.copy()
            self._joints = joints.copy()   # 瞬时到达目标（无动力学）
            self._cmd_count += 1
            self._last_cmd_time = time.perf_counter()
            self._queue_len = 0
        return 0

    def ServoMoveStart(self):
        return 0

    def ServoMoveEnd(self):
        return 0

    def StopMotion(self):
        return 0

    # ------------------------------------------------------------------ #
    #  内部状态读取（供可视化使用）
    # ------------------------------------------------------------------ #

    def get_state(self):
        """返回 (joints, target, cmd_count, last_cmd_time, queue_len) 的快照。"""
        with self._lock:
            return (self._joints.copy(), self._target.copy(),
                    self._cmd_count, self._last_cmd_time, self._queue_len)


def start_visualization(mock: MockFR3, stop_event: threading.Event) -> None:
    """
    在当前线程启动 matplotlib 实时图表（须在主线程调用）。
    图表关闭或 stop_event 被 set 时返回。

    图表布局：
      左侧：水平柱状图，显示 6 个关节当前角度（蓝柱）与目标角度（红点）
      右侧：实时文本面板（指令数、最近指令时间、队列长度）
    """
    fig, (ax_bar, ax_info) = plt.subplots(
        1, 2, figsize=(11, 5),
        gridspec_kw={"width_ratios": [3, 1]}
    )
    fig.canvas.manager.set_window_title("FR3 Simulator -- Alicia-D Teleop")
    fig.patch.set_facecolor("#1a1a2e")

    ax_bar.set_facecolor("#16213e")
    y_pos = np.arange(6)
    bar_h = 0.45

    for i in range(6):
        ax_bar.barh(y_pos[i], JOINT_MAX[i] - JOINT_MIN[i],
                    left=JOINT_MIN[i], height=bar_h + 0.1,
                    color="#0f3460", alpha=0.5, zorder=1)
        ax_bar.axvline(0, color="#444466", linewidth=0.8, zorder=1)

    bars = ax_bar.barh(y_pos, np.zeros(6), height=bar_h,
                        color="#4cc9f0", zorder=3, label="Current")

    target_dots, = ax_bar.plot(np.zeros(6), y_pos, "o",
                                color="#f72585", markersize=9, zorder=4,
                                label="Target")

    ax_bar.set_yticks(y_pos)
    ax_bar.set_yticklabels(JOINT_NAMES, color="white", fontsize=11)
    ax_bar.tick_params(axis="x", colors="#aaaacc")
    ax_bar.set_xlabel("Angle (deg)", color="#aaaacc")
    ax_bar.set_title("FR3 Joint State (Sim)", color="white", fontsize=12, pad=10)
    ax_bar.legend(loc="lower right", facecolor="#222244", labelcolor="white",
                   fontsize=9)

    for spine in ax_bar.spines.values():
        spine.set_edgecolor("#333355")

    ax_info.set_facecolor("#16213e")
    ax_info.set_xticks([])
    ax_info.set_yticks([])
    for spine in ax_info.spines.values():
        spine.set_edgecolor("#333355")
    ax_info.set_title("Status", color="white", fontsize=11, pad=8)
    info_text = ax_info.text(
        0.05, 0.95, "", transform=ax_info.transAxes,
        fontsize=9, va="top", color="#e0e0ff",
        fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#0f3460", alpha=0.7)
    )

    fig.tight_layout(pad=2.0)

    def _update(_frame):
        if stop_event.is_set():
            plt.close(fig)
            return

        joints, target, cmd_count, last_cmd_t, queue_len = mock.get_state()

        # 更新柱状图
        x_min = np.minimum(joints, 0.0)
        widths = joints - x_min
        for bar, xi, wi in zip(bars, x_min, widths):
            bar.set_x(xi)
            bar.set_width(wi if wi != 0 else 1e-9)

        # 颜色：靠近限位时变红
        margin = np.minimum(joints - JOINT_MIN, JOINT_MAX - joints)
        total_range = JOINT_MAX - JOINT_MIN
        ratio = np.clip(margin / (total_range * 0.1 + 1e-9), 0, 1)
        for bar, r in zip(bars, ratio):
            bar.set_color(plt.cm.RdYlGn(0.2 + 0.8 * r))  # type: ignore[attr-defined]

        target_dots.set_xdata(target)

        # 更新 x 轴范围
        all_vals = np.concatenate([joints, target, JOINT_MIN, JOINT_MAX])
        pad = 10
        ax_bar.set_xlim(all_vals.min() - pad, all_vals.max() + pad)

        # 更新信息面板
        age = f"{time.perf_counter() - last_cmd_t:.3f}s" if last_cmd_t else "N/A"
        lines = [""]
        for i, (j, t_) in enumerate(zip(joints, target)):
            lines.append(f"J{i+1}: {j:+8.2f} -> {t_:+8.2f}")
        lines += [
            "",
            f"Cmds:    {cmd_count}",
            f"Last:    {age}",
            f"Queue:   {queue_len}",
        ]
        info_text.set_text("\n".join(lines))

    anim = FuncAnimation(fig, _update, interval=50, cache_frame_data=False)
    plt.show()
    # 窗口关闭后通知主程序停止
    stop_event.set()
