#!/usr/bin/env python3
"""
Alicia-D 示教臂 → FAIRINO FR3 遥操作控制

通过 alicia_d_sdk 读取 Alicia-D 示教臂（Leader Arm）关节角度，
经轴映射、安全限位、平滑处理后，
通过 FAIRINO SDK ServoJ 实时控制 FR3 六轴机械臂。

使用步骤：
    # Step 1: 检查 FR3 状态（无需 FR3，观察当前轴角度）
    python alicia_teleop_fr3.py --no-robot

    # Step 2: 连接真实 FR3（示教器需在自动模式，机器人已上使能）
    python alicia_teleop_fr3.py --robot-ip 192.168.57.2

运动模式（--relative / --absolute）：
    默认【相对模式】：启动时记录示教臂初始角度作为基准，FR3 跟随相对变化量。
        - 无需示教臂零位校准，启动时 FR3 保持原位不动
        - 只要示教臂不动，FR3 就不动
    --absolute：示教臂角度直接叠加到 FR3 初始角度（需先运行 zero_calibration）

FR3 使用前提（在示教器上完成）：
    1. 示教器切换到【自动模式】（通常是钥匙开关）
    2. 机器人【上使能】（伺服上电）
    3. 清除所有故障

轴映射说明（--axis-order 和 --axis-sign）：
    FR3 第 i 个关节 = fr3_init[i] + axis_sign[i] * delta[axis_order[i]]
    delta = leader_current - leader_start（相对模式）
    delta = leader_current（绝对模式，需零位校准）
"""

import argparse
import math
import os
import sys
import threading
import time

import numpy as np

from alicia_d_sdk import create_robot
from alicia_d_sdk.utils import precise_sleep

FAIRINO_SDK_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Spline", "fairino390", "linux")


# ==========================================================================
#  可调参数（调试时直接改这里；命令行 --xxx 仍可覆盖以下默认值）
# ==========================================================================

# ---- 连接配置 ----
PORT                 = ""                 # Alicia-D 串口端口，留空自动查找
GRIPPER_TYPE         = "50mm"             # Alicia-D 夹爪型号
ROBOT_IP             = "192.168.57.2"     # FR3 机械臂 IP
CONNECT_RETRIES      = 5                  # 示教臂连接失败最大重试次数
CONNECT_RETRY_DELAY  = 3.0               # 每次重试等待（秒）

# ---- 控制参数 ----
RATE                 = 125.0             # 控制频率 (Hz)，FR3 ServoJ 最高 125Hz
MAX_STEP             = 30.0              # 每周期最大关节变化量 (°)，30°≈不限速
JUMP_THRESHOLD       = 180.0            # 跳变检测阈值 (°)，超过视为毛刺丢帧
DEAD_ZONE            = 0.005              # 死区 (°)，变化小于此值不下发 ServoJ

# ---- 平滑滤波参数 ----
FILTER_MINCUTOFF     = 2.0              # [1] OneEuro 截止频率 (Hz)，越小越稳越慢，1~5
FILTER_BETA          = 0.05             # [1] OneEuro 速度自适应，0=恒定，0.02~0.1
TREMOR_CUTOFF        = 3.0              # [2] 固定低通截止 (Hz)，截断手部震颤，2~5
SPRING_OMEGA         = 12.0            # 弹簧阻尼固有频率 ωn (rad/s)，越大越跟手，8~20
MAX_ACCEL            = 500.0            # 关节最大加速度 (°/s²)，抑制暴力加速，200~800
MAX_VEL              = 60.0            # 关节最大速度 (°/s)，安全兜底，60~100

# ---- FR3 关节限位 (°) ----
MIN_ANGLE            = [-170, -265, -145, -265, -170, -355]
MAX_ANGLE            = [170,   85,  145,   85,  170,  355]

# ---- 轴映射配置 ----
AXIS_ORDER           = [0, 1, 2, 3, 4, 5]              # FR3 第 i 轴由 Alicia-D 第 order[i] 轴驱动
AXIS_SIGN            = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]  # +1 同向，-1 反向

# ==========================================================================


class OneEuroFilter:
    """OneEuro 自适应低通滤波器（用于示教臂关节角度抖动抑制）。

    静止时低截止频率 → 强力滤除抖动；运动时截止频率随速度升高 → 减少延迟。
    参考: Géry Casiez et al., "1€ Filter", CHI 2012.

    Args:
        n_joints:   关节数
        mincutoff:  最低截止频率 (Hz)，越小静止越平稳但延迟越大，推荐 2~5 Hz
        beta:       速度自适应系数，越大快速运动时延迟越小，推荐 0.05~0.1
        dcutoff:    导数低通截止频率 (Hz)，一般保持默认
    """
    def __init__(self, n_joints: int = 6, mincutoff: float = 3.0,
                 beta: float = 0.07, dcutoff: float = 1.0):
        self.mincutoff = mincutoff
        self.beta = beta
        self.dcutoff = dcutoff
        self._x = None
        self._dx = np.zeros(n_joints)
        self._last_t: float | None = None

    def reset(self) -> None:
        self._x = None
        self._dx[:] = 0.0
        self._last_t = None

    def step(self, z: np.ndarray, t: float) -> np.ndarray:
        z = np.asarray(z, dtype=float)
        if self._x is None:
            self._x = z.copy()
            self._last_t = t
            return self._x.copy()
        dt = max(1e-6, t - self._last_t)
        self._last_t = t
        # 低通滤波导数
        raw_dx = (z - self._x) / dt
        alpha_d = self._alpha(dt, self.dcutoff)
        self._dx = alpha_d * raw_dx + (1.0 - alpha_d) * self._dx
        # 自适应截止频率
        cutoff = self.mincutoff + self.beta * np.abs(self._dx)
        alpha = self._alpha(dt, cutoff)
        self._x = alpha * z + (1.0 - alpha) * self._x
        return self._x.copy()

    @staticmethod
    def _alpha(dt: float, cutoff) -> np.ndarray:
        tau = 1.0 / (2.0 * np.pi * np.asarray(cutoff, dtype=float))
        return dt / (dt + tau)

class LowPassFilter:
    """固定截止频率一阶 IIR 低通滤波器（用于震颤频段抑制）。

    与 OneEuro 串联使用：OneEuro 负责跟手（自适应），本级负责固定截断震颤（8~12 Hz）。
    无论运动快慢，截止频率始终固定，不会被运动速度拉高。

    Args:
        n_joints:   关节数
        cutoff_hz:  截止频率 (Hz)，低于此频率的运动正常通过，推荐 2~4 Hz
    """
    def __init__(self, n_joints: int = 6, cutoff_hz: float = 3.0):
        self.cutoff_hz = cutoff_hz
        self._y = None
        self._last_t: float | None = None

    def reset(self) -> None:
        self._y = None
        self._last_t = None

    def step(self, x: np.ndarray, t: float) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if self._y is None:
            self._y = x.copy()
            self._last_t = t
            return self._y.copy()
        dt = max(1e-6, t - self._last_t)
        self._last_t = t
        tau = 1.0 / (2.0 * np.pi * self.cutoff_hz)
        alpha = dt / (dt + tau)
        self._y = alpha * x + (1.0 - alpha) * self._y
        return self._y.copy()


try:
    from fairino import Robot as FR3Robot
    FAIRINO_AVAILABLE = True
except ImportError:
    try:
        if FAIRINO_SDK_PATH not in sys.path:
            sys.path.insert(0, FAIRINO_SDK_PATH)
        from fairino import Robot as FR3Robot
        FAIRINO_AVAILABLE = True
    except ImportError:
        FAIRINO_AVAILABLE = False
        print(f"[WARN] 未找到 fairino SDK。已尝试路径: {FAIRINO_SDK_PATH}")


def _diagnose_fr3(fr3) -> bool:
    """
    读取 FR3 状态包，打印诊断信息。
    返回 True 表示机器人处于可控状态，False 表示有阻塞性问题。
    """
    try:
        pkg = fr3.robot_state_pkg
        mode_str = "自动" if pkg.robot_mode == 0 else "手动"
        state_map = {1: "停止", 2: "运行中", 3: "暂停", 4: "拖动示教"}
        state_str = state_map.get(pkg.robot_state, f"未知({pkg.robot_state})")
        print(f"[INFO] FR3 运动模式: {mode_str}  运动状态: {state_str}")

        has_fault = (pkg.main_code != 0)
        if has_fault:
            print(f"[WARN] FR3 存在故障码: 主码={pkg.main_code}  子码={pkg.sub_code}")
            print("[INFO] 尝试自动清除错误...")
            ret = fr3.ResetAllError()
            if ret == 0:
                print("[INFO] FR3 错误已清除")
                has_fault = False
            else:
                print(f"[WARN] 自动清除失败 (ret={ret})，请在示教器手动清除故障后重试")

        if pkg.robot_mode == 1:
            print()
            print("=" * 60)
            print("  [ERROR] FR3 当前处于【手动模式】")
            print("  ServoJ / ServoMoveStart 均无法在手动模式下执行")
            print()
            print("  请在示教器上执行以下操作：")
            print("    1. 将模式钥匙开关拨到【自动】位置")
            print("    2. 确认机器人已【上使能】（伺服绿灯亮）")
            print("    3. 清除所有报警/故障")
            print("    4. 重新运行本脚本")
            print("=" * 60)
            print()
            return False

        return not has_fault

    except AttributeError:
        # robot_state_pkg 可能在连接初期尚未收到 UDP 包
        print("[WARN] 状态包尚未就绪，跳过诊断")
        return True


class AliciaTeleopFR3:
    def __init__(self, port, gripper_type, robot_ip, rate,
                 min_angles, max_angles, max_step, jump_threshold,
                 axis_order, axis_sign, no_robot, args=None):

        self.rate = rate
        self.min_angles = np.array(min_angles, dtype=float)
        self.max_angles = np.array(max_angles, dtype=float)
        self.max_step = max_step
        self.jump_threshold = jump_threshold
        self.axis_order = axis_order
        self.axis_sign = np.array(axis_sign, dtype=float)
        self.no_robot = no_robot
        self._relative = not getattr(args, "absolute", False)
        self._leader_start = None   # 首次读取后设置，用于相对模式

        use_filter = not getattr(args, "no_filter", False)
        mincutoff = getattr(args, "filter_mincutoff", FILTER_MINCUTOFF)
        beta = getattr(args, "filter_beta", FILTER_BETA)
        tremor_cutoff = getattr(args, "tremor_cutoff", TREMOR_CUTOFF)
        # 第一级：OneEuro（自适应，保留跟手性）
        self._leader_filter: OneEuroFilter | None = (
            OneEuroFilter(n_joints=6, mincutoff=mincutoff, beta=beta) if use_filter else None
        )
        # 第二级：固定低通（截断震颤频段，不受运动速度影响）
        self._tremor_filter: LowPassFilter | None = (
            LowPassFilter(n_joints=6, cutoff_hz=tremor_cutoff) if use_filter else None
        )

        self.fr3_init_angles = np.zeros(6)
        self.last_fr3_angles = None
        self._last_velocity = None  # 保留（备用），当前由弹簧阻尼接管
        self._sd_pos: np.ndarray | None = None  # 弹簧阻尼当前位置
        self._sd_vel: np.ndarray = np.zeros(6)  # 弹簧阻尼当前速度（°/s）
        self._spring_omega: float = float(getattr(args, "spring_omega", SPRING_OMEGA))
        self._max_accel: float = float(getattr(args, "max_accel", MAX_ACCEL))  # °/s²
        self._max_vel: float = float(getattr(args, "max_vel", MAX_VEL))        # °/s
        self._fr3_ready = False     # ServoMoveStart 是否成功
        self._last_recovery_time = 0.0  # 上次伺服恢复时间戳

        # 连接 Alicia-D 示教臂（含重试，兼容上电慢/USB 初始化延迟）
        connect_retries = getattr(args, "connect_retries", CONNECT_RETRIES)
        connect_retry_delay = getattr(args, "connect_retry_delay", CONNECT_RETRY_DELAY)
        self.leader = None
        for attempt in range(1, connect_retries + 1):
            try:
                print(f"[INFO] 连接 Alicia-D 示教臂... (尝试 {attempt}/{connect_retries})")
                self.leader = create_robot(port=port, gripper_type=gripper_type)
                if self.leader.is_connected():
                    print("[INFO] Alicia-D 示教臂已连接")
                    break
                print("[WARN] 连接后 is_connected() 返回 False")
            except Exception as e:
                print(f"[WARN] 连接失败: {e}")
            if attempt < connect_retries:
                print(f"[INFO] {connect_retry_delay:.0f} 秒后重试，请确认示教臂已上电...")
                time.sleep(connect_retry_delay)
            else:
                raise RuntimeError(
                    "Alicia-D 示教臂连接失败，已重试 " + str(connect_retries) + " 次。\n"
                    "  - 请确认示教臂电源已打开\n"
                    "  - 请确认 USB 线缆连接正常\n"
                    "  - 可用 --connect-retries N --connect-retry-delay S 调整重试参数"
                )

        self._sim_mode = getattr(args, "sim", False) if hasattr(args, "sim") else False

        # 连接 FR3
        self.fr3 = None
        if self._sim_mode:
            from sim_fr3 import MockFR3
            self.fr3 = MockFR3(robot_ip)
            self._fr3_ready = True
        elif no_robot:
            print("[INFO] --no-robot 模式：只打印目标角度，不连接 FR3")
        elif FAIRINO_AVAILABLE:
            try:
                self.fr3 = FR3Robot.RPC(robot_ip)
                print(f"[INFO] 已连接 FR3 机械臂: {robot_ip}")
                _, version = self.fr3.GetSDKVersion()
                print(f"[INFO] FR3 SDK 版本: {version}")

                # 等待 UDP 状态包到达
                time.sleep(0.5)

                # 诊断：读取当前模式和故障码
                ok = _diagnose_fr3(self.fr3)

                # 尝试切换到自动模式（手动模式下会失败，但仍继续诊断）
                ret = self.fr3.Mode(0)
                if ret == 0:
                    print("[INFO] FR3 已切换到自动模式")
                elif ret != 0 and ok:
                    print(f"[INFO] Mode(0) 返回 {ret}（机器人已在自动模式）")

                # 上使能
                ret = self.fr3.RobotEnable(1)
                if ret == 0:
                    print("[INFO] FR3 已上使能")
                else:
                    print(f"[INFO] RobotEnable(1) 返回 {ret}（机器人已处于使能状态）")

                time.sleep(0.3)

                # 读取当前关节角度（阻塞模式，保证数据最新）
                ret, init_angles = self.fr3.GetActualJointPosDegree(0)
                if ret == 0 and isinstance(init_angles, (list, np.ndarray)) and len(init_angles) == 6:
                    self.fr3_init_angles = np.array(init_angles, dtype=float)
                    print(f"[INFO] FR3 当前关节角度: {np.round(self.fr3_init_angles, 2).tolist()}")
                    # 警告：关节若已在极限附近，需先用示教器归位
                    margin = np.minimum(
                        self.fr3_init_angles - self.min_angles,
                        self.max_angles - self.fr3_init_angles
                    )
                    near_limit = np.where(margin < 5.0)[0]
                    if len(near_limit) > 0:
                        joints_str = ", ".join(f"J{i+1}({self.fr3_init_angles[i]:.1f}°)" for i in near_limit)
                        print(f"[WARN] 以下关节距限位不足 5°: {joints_str}")
                        print("[WARN] 建议先通过示教器将机器人移动到安全位置再启动遥操作")
                else:
                    print(f"[WARN] 获取 FR3 初始角度失败 ret={ret}，以零位为基准")

            except Exception as e:
                print(f"[ERROR] 连接 FR3 失败: {e}")
                self.fr3 = None
        else:
            print("[WARN] fairino SDK 不可用，将仅打印目标角度")

        self.last_fr3_angles = self.fr3_init_angles.copy()

    def _map_to_fr3(self, leader_deg: np.ndarray) -> np.ndarray:
        """
        将 Alicia-D 关节角度映射到 FR3 目标角度。

        相对模式（默认）：target[i] = fr3_init[i] + sign[i] * (leader[j] - leader_start[j])
        绝对模式：        target[i] = fr3_init[i] + sign[i] * leader[j]
        """
        if self._relative and self._leader_start is not None:
            delta = leader_deg - self._leader_start
        else:
            delta = leader_deg

        target = self.fr3_init_angles.copy()
        for i in range(6):
            src = self.axis_order[i]
            target[i] += self.axis_sign[i] * delta[src]
        return target

    def _limit(self, angles: np.ndarray) -> np.ndarray:
        return np.clip(angles, self.min_angles, self.max_angles)

    def _smooth(self, target: np.ndarray):
        """
        三级平滑：
        1. 跳变检测：变化超过 jump_threshold → 丢帧（传感器毛刺）
        2. 速度限制：每帧位移不超过 max_step（限制最大速率）
        3. 加速度限制：每帧速度变化不超过 max_step/3（让启停渐进，消除顿挫）
        """
        delta = target - self.last_fr3_angles
        if np.any(np.abs(delta) > self.jump_threshold):
            print(f"[WARN] 关节跳变，跳过本次。最大变化: {np.round(np.abs(delta).max(), 2)}°")
            self._last_velocity = None
            return None

        # 第2层：速度限制
        vel = np.clip(delta, -self.max_step, self.max_step)

        # 第3层：加速度限制（每帧速度变化量 ≤ max_step/3）
        if self._last_velocity is not None:
            max_accel = self.max_step / 3.0
            dv = vel - self._last_velocity
            vel = self._last_velocity + np.clip(dv, -max_accel, max_accel)

        self._last_velocity = vel.copy()
        return self.last_fr3_angles + vel

    def _spring_damper(self, target: np.ndarray) -> np.ndarray | None:
        """临界阻尼弹簧系统：平滑追踪目标位置。

        手部微停顿时速度自然衰减，不硬停；
        手部重新移动时速度平滑加速，不顿挫。
        内置跳变检测保留传感器毛刺防护。
        """
        # 首帧：从机器人当前位置初始化，避免启动时突跳
        if self._sd_pos is None:
            self._sd_pos = self.last_fr3_angles.copy()
            self._sd_vel = np.zeros(6)
            return self._sd_pos.copy()

        # 跳变检测（以弹簧位置为参考，而非 last_fr3_angles）
        if np.any(np.abs(target - self._sd_pos) > self.jump_threshold):
            print(f"[WARN] 目标跳变，跳过本次。最大偏差: "
                  f"{np.round(np.abs(target - self._sd_pos).max(), 2)}°")
            return self._sd_pos.copy()

        dt = 1.0 / self.rate
        # 临界阻尼 (ζ=1) 弹簧更新
        error = target - self._sd_pos
        acc = self._spring_omega ** 2 * error - 2.0 * self._spring_omega * self._sd_vel
        # 加速度上限：防止示教臂忽然大幅抖动引起机械臂瞬间暴力加速
        acc = np.clip(acc, -self._max_accel, self._max_accel)
        self._sd_vel = self._sd_vel + acc * dt
        # 速度上限：安全兜底，弹簧自然减速时不受影响
        self._sd_vel = np.clip(self._sd_vel, -self._max_vel, self._max_vel)
        self._sd_pos = self._sd_pos + self._sd_vel * dt

        # 关节限位，触限时清零对应轴速度防止积累
        clipped = np.clip(self._sd_pos, self.min_angles, self.max_angles)
        at_limit = clipped != self._sd_pos
        self._sd_vel[at_limit] = 0.0
        self._sd_pos = clipped

        return self._sd_pos.copy()

    def _recover_servo(self) -> bool:
        """清除 FR3 报警并重新启动伺服模式。返回是否恢复成功。"""
        now = time.perf_counter()
        if now - self._last_recovery_time < 0.5:
            return False
        self._last_recovery_time = now
        self._fr3_ready = False
        self._last_velocity = None
        self._sd_pos = None   # 恢复后弹簧状态重置，下一帧从实际位置重新初始化
        self._sd_vel = np.zeros(6)
        try:
            self.fr3.ServoMoveEnd()
            r = self.fr3.ResetAllError()
            if r == 0:
                print("[INFO] FR3 报警已清除，正在重新初始化伺服模式...")
            else:
                print(f"[WARN] ResetAllError 返回 {r}，继续尝试重启伺服")
            time.sleep(0.1)
            r = self.fr3.ServoMoveStart()
            if r == 0:
                print("[INFO] ServoMoveStart 重新成功，继续遥操作")
                self._fr3_ready = True
            else:
                print(f"[ERROR] ServoMoveStart 重新失败 (ret={r})，请检查示教器状态")
        except Exception as e:
            print(f"[ERROR] 伺服恢复异常: {e}")
        return self._fr3_ready

    def _send(self, angles: np.ndarray) -> None:
        """发送关节角度到 FR3，死区内不下发指令。"""
        delta = np.abs(angles - self.last_fr3_angles)
        if np.all(delta < DEAD_ZONE):
            return
        angles = np.where(delta < DEAD_ZONE, self.last_fr3_angles, angles)

        if not self._sim_mode:
            print(f"[ACTION] FR3 目标: {np.round(angles, 2).tolist()}")

        if self.fr3 is not None and self._fr3_ready:
            try:
                cmd_t = max(0.008, 1.0 / self.rate)
                ret = self.fr3.ServoJ(angles.tolist(), [0.0, 0.0, 0.0, 0.0],
                                       0.0, 0.0, cmd_t)
                if ret != 0:
                    print(f"[WARN] ServoJ 错误码: {ret}，尝试自动恢复...")
                    self._recover_servo()
                    return
                _, queue_len = self.fr3.GetMotionQueueLength()
                if queue_len >= 2:
                    print(f"[WARN] 运动队列积压: {queue_len}，可能有延迟")
                    time.sleep(0.008)
            except Exception as e:
                print(f"[ERROR] FR3 控制异常: {e}")

        self.last_fr3_angles = angles.copy()

    def run(self) -> None:
        mode_str = "相对（推荐）" if self._relative else "绝对（需零位校准）"
        print("=" * 60)
        print("    Alicia-D → FR3 遥操作系统已启动")
        print("=" * 60)
        print(f"  控制频率: {self.rate} Hz  最大步长: {self.max_step}°")
        print(f"  运动模式: {mode_str}")
        print(f"  轴顺序:   {self.axis_order}")
        print(f"  轴方向:   {self.axis_sign.tolist()}")
        print("  按 Ctrl+C 安全退出")
        print("-" * 60)

        interval = 1.0 / self.rate
        spin_threshold = 0.002 if interval <= 0.010 else 0.010

        # ServoJ 必须在 ServoMoveStart 之后才能生效
        if self.fr3 is not None and not self._sim_mode:
            ret = self.fr3.ServoMoveStart()
            if ret == 0:
                print("[INFO] FR3 伺服模式已开启 (ServoMoveStart OK)")
                self._fr3_ready = True
            else:
                print(f"[ERROR] ServoMoveStart 失败 (ret={ret})")
                print("[ERROR] 请先在示教器上将机器人切换到【自动模式】并上使能")
                self._fr3_ready = False

        if self._relative:
            print("[INFO] 相对模式：保持示教臂静止，FR3 将不动。移动示教臂后 FR3 跟随。")

        try:
            while True:
                t0 = time.perf_counter()

                try:
                    state = self.leader.get_robot_state("joint_gripper", timeout=0.1)
                except Exception:
                    continue
                if state is None:
                    continue

                leader_deg = np.array(state.angles) * (180.0 / math.pi)

                # 第一级：OneEuro（自适应，静止防抖 + 运动跟手）
                if self._leader_filter is not None:
                    leader_deg = self._leader_filter.step(leader_deg, t0)
                # 第二级：固定低通（截断手部震颤 8~12 Hz，不受速度影响）
                if self._tremor_filter is not None:
                    leader_deg = self._tremor_filter.step(leader_deg, t0)

                # 相对模式：记录首次读数为起点
                if self._relative and self._leader_start is None:
                    self._leader_start = leader_deg.copy()
                    print(f"[INFO] 示教臂起始位置已记录: {np.round(self._leader_start, 1).tolist()}")
                    print("[INFO] 现在移动示教臂，FR3 将跟随相对变化量")

                target = self._map_to_fr3(leader_deg)
                target = self._limit(target)
                smoothed = self._spring_damper(target)
                if smoothed is None:
                    continue

                self._send(smoothed)

                precise_sleep(max(0.0, interval - (time.perf_counter() - t0)),
                               spin_threshold=spin_threshold)

        except KeyboardInterrupt:
            print("\n[STOP] 收到中断信号，正在安全停止...")
        finally:
            if self.fr3 is not None:
                try:
                    self.fr3.ServoMoveEnd()
                    self.fr3.StopMotion()
                    print("[INFO] FR3 伺服模式已关闭，运动已停止")
                except Exception as e:
                    print(f"[WARN] 停止 FR3 时出错: {e}")
            self.leader.disconnect()
            print("[INFO] 示教臂已断开，程序退出")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Alicia-D 示教臂 → FAIRINO FR3 遥操作控制",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # 连接配置（默认值见文件顶部 CONFIG 区）
    parser.add_argument("--port", type=str, default=PORT,
                         help="Alicia-D 串口端口（留空自动查找）")
    parser.add_argument("--gripper-type", type=str, default=GRIPPER_TYPE,
                         help="Alicia-D 夹爪型号")
    parser.add_argument("--robot-ip", type=str, default=ROBOT_IP,
                         help="FR3 机械臂 IP 地址")
    parser.add_argument("--no-robot", action="store_true",
                         help="不连接 FR3，只打印目标角度（验证轴映射用）")
    parser.add_argument("--sim", action="store_true",
                         help="仿真模式：用 MockFR3 替代真实 FR3，打开 matplotlib 实时图表")
    parser.add_argument("--connect-retries", type=int, default=CONNECT_RETRIES,
                         help="示教臂连接失败时的最大重试次数")
    parser.add_argument("--connect-retry-delay", type=float, default=CONNECT_RETRY_DELAY,
                         help="每次重试之间的等待时间（秒）")

    # 控制参数
    parser.add_argument("--rate", type=float, default=RATE,
                         help="控制频率 (Hz)，FR3 ServoJ 支持最高 125Hz")
    parser.add_argument("--max-step", type=float, default=MAX_STEP,
                         help="每控制周期最大关节角度变化量（度）。默认 30° 等效于不限速，FR3 物理极限远低于此")
    parser.add_argument("--jump-threshold", type=float, default=JUMP_THRESHOLD,
                         help="跳变检测阈值（度），超过此值认为是传感器毛刺，跳过本次")

    # OneEuro 滤波器参数
    parser.add_argument("--no-filter", action="store_true",
                         help="禁用 OneEuro 滤波器，回退到仅步长限制")
    parser.add_argument("--filter-mincutoff", type=float, default=FILTER_MINCUTOFF,
                         help="OneEuro 截止频率 (Hz)。越小抑制抖动越强但响应越慢；建议范围 1~5")
    parser.add_argument("--filter-beta", type=float, default=FILTER_BETA,
                         help="OneEuro 速度自适应系数。0=恒定滤波；>0 时运动越快滤波越弱响应越快，推荐范围 0.02~0.1")
    parser.add_argument("--tremor-cutoff", type=float, default=TREMOR_CUTOFF,
                         help="第二级固定低通截止频率 (Hz)，专门截断手部震颤（8~12 Hz），推荐范围 2~5 Hz")
    parser.add_argument("--spring-omega", type=float, default=SPRING_OMEGA,
                         help="弹簧阻尼固有频率 ωn (rad/s)。越大跟随越快越跟手，越小启停越柔和，推荐范围 8~20")
    parser.add_argument("--max-accel", type=float, default=MAX_ACCEL,
                         help="关节最大加速度 (°/s²)。限制示教臂抖动引起的暴力加速，推荐范围 200~800")
    parser.add_argument("--max-vel", type=float, default=MAX_VEL,
                         help="关节最大速度 (°/s)。安全兜底上限，FR3 额定关节速度约 100°/s，推荐范围 60~100")

    # FR3 关节限位（度）
    parser.add_argument("--min-angle", type=float, nargs=6,
                         default=MIN_ANGLE,
                         help="FR3 各关节最小角度（度）")
    parser.add_argument("--max-angle", type=float, nargs=6,
                         default=MAX_ANGLE,
                         help="FR3 各关节最大角度（度）")

    # 轴映射配置
    parser.add_argument("--axis-order", type=int, nargs=6,
                         default=AXIS_ORDER,
                         help="轴映射顺序：axis-order[i]=j 表示 FR3 第 i 轴由 Alicia-D 第 j 轴驱动")
    parser.add_argument("--axis-sign", type=float, nargs=6,
                         default=AXIS_SIGN,
                         help="轴方向（每 FR3 关节一个值）：+1 同向，-1 反向")

    # 运动模式
    parser.add_argument("--absolute", action="store_true",
                         help="绝对模式：示教臂角度直接叠加到 FR3 初始角度（需示教臂已零位校准）。"
                              "默认为相对模式：仅跟随示教臂相对起始位置的变化量")

    args = parser.parse_args()

    print("=" * 60)
    print("    Alicia-D → FR3 遥操作控制系统")
    print("=" * 60)
    print(f"  Alicia-D 串口: {args.port or '自动检测'}")
    print(f"  FR3 IP:        {args.robot_ip}")
    print(f"  连接机械臂:    {'否（仅打印）' if args.no_robot else '是'}")
    print(f"  控制频率:      {args.rate} Hz")
    print(f"  最大步长:      {args.max_step}°")
    if args.no_filter:
        print(f"  平滑滤波:      关闭（仅步长限制）")
    else:
        print(f"  平滑滤波:      [1] OneEuro  mincutoff={args.filter_mincutoff} Hz  beta={args.filter_beta}")
        print(f"                 [2] 固定低通  cutoff={args.tremor_cutoff} Hz（震颤抑制）")
    print(f"  运动模式:      {'绝对' if args.absolute else '相对（默认）'}")
    print(f"  轴映射顺序:    {args.axis_order}")
    print(f"  轴方向:        {args.axis_sign}")
    print("-" * 60)

    teleop = AliciaTeleopFR3(
        port=args.port,
        gripper_type=args.gripper_type,
        robot_ip=args.robot_ip,
        rate=args.rate,
        min_angles=args.min_angle,
        max_angles=args.max_angle,
        max_step=args.max_step,
        jump_threshold=args.jump_threshold,
        axis_order=args.axis_order,
        axis_sign=args.axis_sign,
        no_robot=args.no_robot,
        args=args,
    )

    if args.sim:
        from sim_fr3 import start_visualization
        stop_event = threading.Event()
        loop_thread = threading.Thread(target=teleop.run, daemon=True)
        loop_thread.start()
        start_visualization(teleop.fr3, stop_event)
        stop_event.set()
    else:
        teleop.run()
