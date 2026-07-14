#!/usr/bin/env python3
"""
Alicia-D 示教臂 → FAIRINO FR3 遥操作控制

通过 alicia_d_sdk 读取 Alicia-D 示教臂（Leader Arm）关节角度，
经轴映射、安全限位、平滑处理后，
通过 FAIRINO SDK ServoJ 实时控制 FR3 六轴机械臂；
同一循环里还会读取主臂上靠近 J6 的夹爪扳机，把它当成"J7"一起走同一套
OneEuro + 固定低通滤波（共用降噪流水线，减少手部震颤对夹爪开合的影响），
再用 MIT 力矩模式 + 软件虚拟弹簧驱动 Gloria-M 从臂夹爪，扭矩限幅在
±gripper-tau-max，避免夹到硬物体时电流无限增大导致过流保护跳闸。

使用步骤：
    # Step 1: 检查 FR3 状态（无需 FR3，观察当前轴角度）
    python alicia_teleop_fr3.py --no-robot --no-gripper

    # Step 2: 连接真实 FR3 + 夹爪（示教器需在自动模式，机器人已上使能）
    python alicia_teleop_fr3.py --robot-ip 192.168.57.2 --gripper-port /dev/ttyACM1

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
from collections import deque

import numpy as np

from alicia_d_sdk import create_robot
from alicia_d_sdk.utils import precise_sleep

from gloria_m_sdk import ControlMode, GloriaGripper, Limits, PositionRange, Variable

FAIRINO_SDK_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Spline", "fairino390", "linux")


def _clamp(value: float, low: float, high: float) -> float:
    if value < low:
        return low
    if value > high:
        return high
    return value


def _ensure_gripper_kt_value(g: GloriaGripper) -> None:
    """若电机的 KT_Value（转矩常数）寄存器为 0，MIT 模式会被固件拒绝确认
    （电流环算不出该给多少电流）。这里现场按标准 PMSM 公式 Kt=1.5*NPP*Flux
    估算一个候选值，临时写入（不调用 save()，断电/下次上电即恢复为 0）。

    这是标定值缺失时的应急估算，不是厂家标定的准确值；后续如果厂家提供了
    准确的 KT_Value，应改为在标定流程里正式写入并 save()。
    """
    rid_kt = int(Variable.KT_Value)
    current = g.params.read(rid_kt, timeout_s=0.2)
    if current is not None and abs(current) > 1e-9:
        return  # 已经有非零值（可能厂家已标定），不覆盖

    npp = g.params.read(int(Variable.NPP), timeout_s=0.2)
    flux = g.params.read(int(Variable.Flux), timeout_s=0.2)
    if npp is None or flux is None:
        print("[WARN] 无法读取 NPP/Flux，跳过 KT_Value 估算；MIT 模式切换可能会失败")
        return

    candidate = 1.5 * float(npp) * float(flux)
    g.params.write_f32(rid_kt, candidate)
    print(f"[WARN] 检测到夹爪 KT_Value=0（未标定），已临时写入估算值 {candidate:.6f} Nm/A"
          f"（仅本次通电有效，未写入 Flash；这是应急估算，不是厂家标定值）")


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
FILTER_MINCUTOFF     = 3.0              # [1] OneEuro 截止频率 (Hz)，越小越稳越慢，1~5
FILTER_BETA          = 0.05             # [1] OneEuro 速度自适应，0=恒定，0.02~0.1
TREMOR_CUTOFF        = 5.0              # [2] 固定低通截止 (Hz)，截断手部震颤，2~5（为降延迟从3上调，震颤抑制随之减弱）
SPRING_OMEGA         = 20.0            # 弹簧阻尼固有频率 ωn (rad/s)，越大越跟手，8~20（标准值：12）
MAX_ACCEL            = 800.0            # 关节最大加速度 (°/s²)，抑制暴力加速，200~800（标准值：500）
MAX_VEL              =150           # 关节最大速度 (°/s)，安全兜底，60~100 （标准值：60）
PREDICT_ENABLED       = True   # 是否启用前瞻预测（接在滤波+弹簧之后，对干净的弹簧速度做线性外推，抵消上游延迟）
PREDICT_LOOKAHEAD_MS  = 10.0   # 前瞻时长 (ms)。从小值开始测试，逐步上调，若出现超调/震颤则调小

# ---- FR3 关节限位 (°) ----
MIN_ANGLE            = [-170, -265, -145, -265, -170, -355]
MAX_ANGLE            = [170,   85,  145,   85,  170,  355]

# ---- 轴映射配置 ----
AXIS_ORDER           = [0, 1, 2, 3, 4, 5]              # FR3 第 i 轴由 Alicia-D 第 order[i] 轴驱动
AXIS_SIGN            = [1.0, 1.0, 1.0, 1.0, -1.0, -1.0]  # +1 同向，-1 反向

# ---- 夹爪（Gloria-M）连接配置 ----
GRIPPER_PORT         = "auto"    # 从臂夹爪串口号；'auto' 自动检测唯一可用端口
GRIPPER_BAUD         = 921_600
GRIPPER_CMD_ID       = 0x01
GRIPPER_FB_ID        = 0x101

# ---- 从臂夹爪位置范围（弧度，未标定，先用占位值）----
GRIPPER_OPEN_Q       = 2.5       # 从臂全开对应角度 [rad]（占位，需标定）
GRIPPER_CLOSE_Q      = 0.0       # 从臂全合对应角度 [rad]（占位，需标定）

# ---- 夹爪力/电流限制（MIT 力矩模式核心参数，详见 gripper_teleop.py 里的标定说明）----
GRIPPER_TAU_MAX      = 0.75      # 最大闭合/张开扭矩 [Nm]，需按电源电流表边测边调
GRIPPER_STIFFNESS    = 6.0       # 虚拟弹簧刚度 [Nm/rad]
GRIPPER_KD           = 0.15      # MIT 阻尼 [Nm·s/rad]

# ---- 夹爪原始值(0~1000)缩放系数 ----
# 主臂扳机被当成"J7"跟 6 个关节角度(单位:度)一起走同一套 OneEuro/固定低通滤波，
# 但扳机原始量程 0~1000 比关节角度(几十度量级)大得多；除以这个系数把它压缩到
# 相近的数量级，这样两个滤波器共用的 mincutoff/beta/tremor-cutoff 才有意义，
# 滤波之后再乘回去还原成 0~1000 量程。
GRIPPER_RAW_SCALE    = 100.0

# ---- 夹爪 MIT 协议编码范围（tau 实际上限由 GRIPPER_TAU_MAX 控制）----
GRIPPER_LIMITS_PMAX  = 3.14
GRIPPER_LIMITS_VMAX  = 10.0
GRIPPER_LIMITS_TMAX  = 6.0

# ---- 端到端延迟埋点 ----
LATENCY_PROBE          = True   # 是否打印"主臂原始位移→对应指令发出"的实测延迟
LATENCY_PROBE_THRESHOLD = 2.0   # 触发/响应阈值 (°)，越小越灵敏但越容易被噪声误触发
LATENCY_PROBE_QUIESCENT = 0.3   # 静止判定阈值 (°)，约200ms窗口内总位移小于此值才更新基准

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

        # ---- 夹爪（Gloria-M）配置 ----
        self._gripper_open_q = float(getattr(args, "gripper_open_q", GRIPPER_OPEN_Q))
        self._gripper_close_q = float(getattr(args, "gripper_close_q", GRIPPER_CLOSE_Q))
        self._gripper_tau_max = float(getattr(args, "gripper_tau_max", GRIPPER_TAU_MAX))
        self._gripper_stiffness = float(getattr(args, "gripper_stiffness", GRIPPER_STIFFNESS))
        self._gripper_kd = float(getattr(args, "gripper_kd", GRIPPER_KD))
        self._gripper_raw_scale = GRIPPER_RAW_SCALE
        self.gripper: GloriaGripper | None = None

        use_filter = not getattr(args, "no_filter", False)
        mincutoff = getattr(args, "filter_mincutoff", FILTER_MINCUTOFF)
        beta = getattr(args, "filter_beta", FILTER_BETA)
        tremor_cutoff = getattr(args, "tremor_cutoff", TREMOR_CUTOFF)
        # 第一级：OneEuro（自适应，保留跟手性）。n_joints=7：6 个机械臂关节
        # + 1 个夹爪扳机（当成"J7"跟关节角度共用同一套滤波流水线）。
        self._leader_filter: OneEuroFilter | None = (
            OneEuroFilter(n_joints=7, mincutoff=mincutoff, beta=beta) if use_filter else None
        )
        # 第二级：固定低通（截断震颤频段，不受运动速度影响）
        self._tremor_filter: LowPassFilter | None = (
            LowPassFilter(n_joints=7, cutoff_hz=tremor_cutoff) if use_filter else None
        )

        self.fr3_init_angles = np.zeros(6)
        self.last_fr3_angles = None
        self._last_velocity = None  # 保留（备用），当前由弹簧阻尼接管
        self._sd_pos: np.ndarray | None = None  # 弹簧阻尼当前位置
        self._sd_vel: np.ndarray = np.zeros(6)  # 弹簧阻尼当前速度（°/s）
        self._last_target: np.ndarray | None = None  # 上一次被接受的逐关节目标（用于逐关节跳变检测）
        self._leader_prev_raw: np.ndarray | None = None  # 上一帧主臂原始角（度），用于解绕
        self._leader_unwrap_offset = np.zeros(6)  # 累积解绕偏移（±360 的整数倍）
        self._latency_probe = bool(getattr(args, "latency_probe", LATENCY_PROBE))
        self._latency_threshold = float(getattr(args, "latency_probe_threshold", LATENCY_PROBE_THRESHOLD))
        self._latency_baseline: np.ndarray | None = None  # 主臂静止基准（原始角度，逐轴）
        self._latency_history: deque = deque(maxlen=max(1, int(0.2 * rate)))  # 约200ms窗口，用于判定是否静止
        self._latency_active = np.zeros(6, dtype=bool)     # 该轴是否正在等待响应
        self._latency_onset_t = np.zeros(6)                # 触发时刻
        self._latency_onset_sent = np.zeros(6)             # 触发时刻已发送的角度
        self._predict_enabled = bool(getattr(args, "predict_enabled", PREDICT_ENABLED))
        self._predict_lookahead_ms = float(getattr(args, "predict_lookahead_ms", PREDICT_LOOKAHEAD_MS))
        self._spring_omega: float = float(getattr(args, "spring_omega", SPRING_OMEGA))
        self._max_accel: float = float(getattr(args, "max_accel", MAX_ACCEL))  # °/s²
        self._max_vel: float = float(getattr(args, "max_vel", MAX_VEL))        # °/s
        self._send_frame_count = 0  # _send() 调用计数，用于降频查询运动队列长度
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

                # 诊断：读取当前模式和故障码（有打印/清错副作用）
                _diagnose_fr3(self.fr3)

                # 切换到自动模式：仅在当前为手动模式时才调用。
                # 已在自动模式时重复调用 Mode(0) 在带遗留暂停/刚 ResetAllError 时会阻塞卡死，
                # 故先读状态包判断，已是自动就跳过。
                try:
                    already_auto = (self.fr3.robot_state_pkg.robot_mode == 0)
                except AttributeError:
                    already_auto = False
                if already_auto:
                    print("[INFO] FR3 已处于自动模式（跳过 Mode(0)）")
                else:
                    print("[INFO] 切换到自动模式 Mode(0)...")
                    ret = self.fr3.Mode(0)
                    print(f"[INFO] Mode(0) 返回 {ret}"
                          + ("（已切换到自动模式）" if ret == 0 else ""))

                # 上使能
                print("[INFO] 上使能 RobotEnable(1)...")
                ret = self.fr3.RobotEnable(1)
                if ret == 0:
                    print("[INFO] FR3 已上使能")
                else:
                    print(f"[INFO] RobotEnable(1) 返回 {ret}（机器人已处于使能状态）")

                time.sleep(0.3)

                # 读取当前关节角度（阻塞模式，保证数据最新）
                print("[INFO] 读取当前关节角度 GetActualJointPosDegree(0)...")
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

        # ---- 连接从臂夹爪（Gloria-M）----
        # 失败不影响机械臂遥操作：打印警告后继续，self.gripper 保持 None，
        # run() 里逐帧判空跳过夹爪相关逻辑。
        no_gripper = getattr(args, "no_gripper", False)
        if no_gripper:
            print("[INFO] --no-gripper：不连接从臂夹爪")
        else:
            gripper_port = getattr(args, "gripper_port", GRIPPER_PORT)
            try:
                if gripper_port and gripper_port.lower() != "auto":
                    resolved_port = gripper_port
                else:
                    from serial.tools import list_ports
                    found = list(list_ports.comports())
                    if len(found) != 1:
                        raise RuntimeError(
                            f"串口自动检测失败（找到 {len(found)} 个），请用 --gripper-port 显式指定"
                        )
                    resolved_port = found[0].device

                gripper_limits = Limits(pmax=GRIPPER_LIMITS_PMAX, vmax=GRIPPER_LIMITS_VMAX,
                                        tmax=GRIPPER_LIMITS_TMAX)
                gripper_safe_q = PositionRange(
                    min=min(self._gripper_open_q, self._gripper_close_q),
                    max=max(self._gripper_open_q, self._gripper_close_q),
                )
                print(f"[INFO] 连接从臂夹爪 Gloria-M ({resolved_port})...")
                self.gripper = GloriaGripper(
                    resolved_port,
                    baudrate=getattr(args, "gripper_baud", GRIPPER_BAUD),
                    command_id=getattr(args, "gripper_cmd_id", GRIPPER_CMD_ID),
                    feedback_id=getattr(args, "gripper_fb_id", GRIPPER_FB_ID),
                    limits=gripper_limits,
                    safe_position=gripper_safe_q,
                )
                self.gripper.connect()
                _ensure_gripper_kt_value(self.gripper)
                self.gripper.motor.set_mode(ControlMode.MIT)
                self.gripper.motor.enable()
                self.gripper.motor.refresh()
                print(f"[INFO] 从臂夹爪已连接并上使能，当前位置: {self.gripper.state.position:.3f} rad")
            except Exception as e:
                print(f"[WARN] 从臂夹爪连接失败，本次运行不控制夹爪: {e}")
                if self.gripper is not None:
                    try:
                        self.gripper.disconnect()
                    except Exception:
                        pass
                self.gripper = None

    def _unwrap_leader(self, raw_deg: np.ndarray) -> np.ndarray:
        """对主臂关节角做连续化解绕。

        revolute 关节读数跨越 ±180° 边界时会瞬间跳 ~360°，映射后表现为目标假跳变。
        这种跳变是持续性的（人手停在该区域时每帧都存在），若不在源头消除，
        会让下游的跳变检测长期判定为"毛刺"、永久卡在旧目标上。这里检测相邻帧
        >180° 的突变，累积补偿 ∓360°，使关节角变为连续信号（无回绕时为恒等操作）。
        """
        if self._leader_prev_raw is not None:
            diff = raw_deg - self._leader_prev_raw
            self._leader_unwrap_offset[diff > 180.0] -= 360.0
            self._leader_unwrap_offset[diff < -180.0] += 360.0
        self._leader_prev_raw = raw_deg.copy()
        return raw_deg + self._leader_unwrap_offset

    def _map_to_fr3(self, leader_deg: np.ndarray) -> np.ndarray:
        """
        将 Alicia-D 关节角度映射到 FR3 目标角度。

        相对模式（默认）：target[i] = fr3_init[i] + sign[i] * (leader[j] - leader_start[j])
        绝对模式：        target[i] = fr3_init[i] + sign[i] * leader[j]

        离合式限位（仅相对模式）：某轴目标超出限位时，持续将该轴的 leader_start
        重新校准，使目标恰好钉在边界上，不再累积"虚位"。这样主臂一旦反向，
        哪怕只反转一点，目标立刻离开边界反向移动，无需先转回同样的越界距离
        才能重新触发跟随（避免了积分饱和/windup 式的死区）。
        """
        if self._relative and self._leader_start is not None:
            delta = leader_deg - self._leader_start
        else:
            delta = leader_deg

        target = self.fr3_init_angles.copy()
        for i in range(6):
            src = self.axis_order[i]
            target[i] += self.axis_sign[i] * delta[src]

        if self._relative and self._leader_start is not None:
            for i in range(6):
                src = self.axis_order[i]
                if target[i] > self.max_angles[i]:
                    bound = self.max_angles[i]
                elif target[i] < self.min_angles[i]:
                    bound = self.min_angles[i]
                else:
                    continue
                self._leader_start[src] = leader_deg[src] - self.axis_sign[i] * (bound - self.fr3_init_angles[i])
                target[i] = bound

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

        跳变检测逐关节独立：
        - 比较基准是「本帧 target 相对上一次被接受的 target」（帧间连续性），
          而非「target 相对弹簧当前位置」——后者会把关节到达限位/被 MAX_VEL
          限速导致的合法滞后误判为毛刺，且冻结是全体六轴一起冻结。
        - 判定为毛刺的关节，本帧 target 替换为上一次接受的值（该轴本帧不追新目标），
          其余关节的 target 正常参与积分，互不影响。
        - 前提：主臂角度已在 run() 中经 _unwrap_leader 解绕，真实回绕不会以
          "持续性大跳变"的形式出现在这里，本检测只需处理真正的瞬时单帧毛刺。
        """
        # 首帧：从机器人当前位置初始化，避免启动时突跳
        if self._sd_pos is None:
            self._sd_pos = self.last_fr3_angles.copy()
            self._sd_vel = np.zeros(6)
            self._last_target = target.copy()
            return self._sd_pos.copy()

        # 逐关节跳变检测：仅冻结触发的关节，其余关节的目标不受影响
        frame_delta = np.abs(target - self._last_target)
        glitch = frame_delta > self.jump_threshold
        if np.any(glitch):
            joints = ", ".join(
                f"J{i + 1}({frame_delta[i]:.1f}°)" for i in np.where(glitch)[0]
            )
            print(f"[WARN] 关节目标跳变，仅冻结该轴本帧: {joints}")
            target = np.where(glitch, self._last_target, target)
        self._last_target = target.copy()

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

    def _predict_ahead(self, sd_pos: np.ndarray, sd_vel: np.ndarray) -> np.ndarray:
        """前瞻预测：对弹簧阻尼输出做线性外推，抵消上游（两级滤波+弹簧）已产生的延迟。

        只在弹簧阻尼"干净"的速度状态（已被 MAX_VEL 限幅、经临界阻尼平滑）上做外推，
        不对主臂原始信号做外推——避免把尚未压制的手部震颤噪声重新放大。
        外推结果仍裁剪到关节限位，防止预测把指令推过物理边界。
        """
        lookahead_s = self._predict_lookahead_ms / 1000.0
        predicted = sd_pos + sd_vel * lookahead_s
        return np.clip(predicted, self.min_angles, self.max_angles)

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
                # 队列长度查询是纯诊断用途、与本帧下发的角度无关，降频到每 25 帧
                # 一次（125Hz 下约 5 次/秒），避免每帧多打一次 RPC 拖慢控制循环；
                # 且不再用 sleep 阻塞主循环——阻塞只会延迟下一帧指令，无助于排空队列。
                self._send_frame_count += 1
                if self._send_frame_count % 25 == 0:
                    _, queue_len = self.fr3.GetMotionQueueLength()
                    if queue_len >= 2:
                        print(f"[WARN] 运动队列积压: {queue_len}，可能有延迟")
            except Exception as e:
                print(f"[ERROR] FR3 控制异常: {e}")

        self.last_fr3_angles = angles.copy()

    def _measure_latency(self, leader_raw: np.ndarray, t_now: float) -> None:
        """端到端延迟埋点。

        测量"主臂原始角度（滤波前）产生明显位移"到"对应关节的 ServoJ 实际
        发出角度跟上同等位移"之间的时间差，覆盖整条平滑流水线（两级滤波、
        弹簧阻尼、死区）的真实生效延迟，而非纯理论群延迟估算。

        逐关节独立触发/清零：以 leader_raw（Alicia-D 轴序）检测触发，以
        self.last_fr3_angles（FR3 轴序，_send 实际下发后的值）检测响应，
        通过 axis_order 对应同一物理关节，幅值不受 axis_sign 正负号影响
        （比较时取绝对值）。

        基准更新采用约200ms时间窗判定"是否静止"（而非单帧变化量）：窗口内
        总位移小于静止阈值才更新基准；一旦开始运动（哪怕逐帧变化很小的慢速
        持续运动）基准立即冻结，让累积位移正确地涨过触发阈值——单帧判定会
        被慢速持续运动的微小逐帧增量骗过，永远追平基准、无法触发。

        注意：本方法测量的是"位移达到阈值"所需时间，对缓慢持续拖动会把
        "手本身移动这么远耗费的时间"也计入，导致数字虚高；只有对相对
        快速、干脆的单次动作，测出的数字才能真实反映流水线本身的延迟。
        """
        self._latency_history.append(leader_raw.copy())
        if self._latency_baseline is None:
            self._latency_baseline = leader_raw.copy()
            return
        if len(self._latency_history) < self._latency_history.maxlen:
            return
        window_start = self._latency_history[0]
        for i in range(6):
            src = self.axis_order[i]
            if not self._latency_active[i]:
                if abs(leader_raw[src] - window_start[src]) < LATENCY_PROBE_QUIESCENT:
                    self._latency_baseline[src] = leader_raw[src]
                if abs(leader_raw[src] - self._latency_baseline[src]) > self._latency_threshold:
                    self._latency_active[i] = True
                    self._latency_onset_t[i] = t_now
                    self._latency_onset_sent[i] = self.last_fr3_angles[i]
            else:
                if abs(self.last_fr3_angles[i] - self._latency_onset_sent[i]) > self._latency_threshold:
                    latency_ms = (t_now - self._latency_onset_t[i]) * 1000.0
                    print(f"[延迟埋点] J{i + 1} 端到端延迟: {latency_ms:.1f} ms")
                    self._latency_active[i] = False
                    self._latency_baseline[src] = leader_raw[src]
        self._latency_prev_raw = leader_raw.copy()

    def _drive_gripper(self, gripper_raw: float) -> None:
        """把(滤波后的)主臂夹爪开合值(0~1000)映射到从臂目标角度，
        用软件虚拟弹簧算扭矩前馈并限幅在 ±tau_max，再以 MIT 模式下发。

        自由空间里表现接近位置跟随；夹到硬物体后位置误差持续增大，
        但 tau 已经封顶——闭合力矩（进而电流/功率）不会再涨，
        不会因为过流保护跳闸而失能。详见 gripper_teleop.py 里的标定说明。
        """
        frac = _clamp(gripper_raw / 1000.0, 0.0, 1.0)
        target_q = self._gripper_close_q + frac * (self._gripper_open_q - self._gripper_close_q)

        error = target_q - self.gripper.state.position
        tau_cmd = _clamp(self._gripper_stiffness * error, -self._gripper_tau_max, self._gripper_tau_max)

        try:
            self.gripper.motion.send_mit(kp=0.0, kd=self._gripper_kd, q=0.0, dq=0.0,
                                          tau=tau_cmd, poll=True)
        except Exception as e:
            print(f"[WARN] 夹爪控制异常，本帧跳过: {e}")

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
                # 解绕：消除 ±360° 回绕造成的持续性假跳变（必须在滤波前，否则滤的是不连续信号）
                leader_deg = self._unwrap_leader(leader_deg)
                leader_raw = leader_deg.copy()  # 滤波前的原始角度，用于延迟埋点

                # 夹爪扳机(0~1000)缩放到跟关节角度(度)相近的量级，当成"J7"
                # 跟 6 个关节角度拼在一起，走同一套 OneEuro + 固定低通滤波。
                gripper_raw = float(state.gripper) / self._gripper_raw_scale
                combined = np.concatenate([leader_deg, [gripper_raw]])

                # 第一级：OneEuro（自适应，静止防抖 + 运动跟手）
                if self._leader_filter is not None:
                    combined = self._leader_filter.step(combined, t0)
                # 第二级：固定低通（截断手部震颤 8~12 Hz，不受速度影响）
                if self._tremor_filter is not None:
                    combined = self._tremor_filter.step(combined, t0)

                leader_deg = combined[:6]
                gripper_smoothed_raw = combined[6] * self._gripper_raw_scale

                # 相对模式：记录首次读数为起点
                if self._relative and self._leader_start is None:
                    self._leader_start = leader_deg.copy()
                    print(f"[INFO] 示教臂起始位置已记录: {np.round(self._leader_start, 1).tolist()}")
                    print("[INFO] 现在移动示教臂，FR3 将跟随相对变化量")

                if self.gripper is not None:
                    self._drive_gripper(gripper_smoothed_raw)

                target = self._map_to_fr3(leader_deg)
                target = self._limit(target)
                smoothed = self._spring_damper(target)
                if smoothed is None:
                    continue
                to_send = self._predict_ahead(smoothed, self._sd_vel) if self._predict_enabled else smoothed

                self._send(to_send)
                if self._latency_probe:
                    self._measure_latency(leader_raw, t0)

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
            if self.gripper is not None:
                try:
                    self.gripper.motor.disable()
                    print("[INFO] 从臂夹爪已失能")
                except Exception as e:
                    print(f"[WARN] 夹爪失能时出错: {e}")
                finally:
                    self.gripper.disconnect()
                    print("[INFO] 从臂夹爪已断开")
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

    # 夹爪（Gloria-M）连接与力控参数
    parser.add_argument("--no-gripper", action="store_true",
                         help="不连接从臂夹爪，只控制机械臂")
    parser.add_argument("--gripper-port", type=str, default=GRIPPER_PORT,
                         help="Gloria-M 串口号；'auto' 自动检测唯一可用端口")
    parser.add_argument("--gripper-baud", type=int, default=GRIPPER_BAUD, help="夹爪串口波特率")
    parser.add_argument("--gripper-cmd-id", type=lambda s: int(s, 0), default=GRIPPER_CMD_ID,
                         help="夹爪电机命令 CAN ID")
    parser.add_argument("--gripper-fb-id", type=lambda s: int(s, 0), default=GRIPPER_FB_ID,
                         help="夹爪电机反馈 CAN ID")
    parser.add_argument("--gripper-open-q", type=float, default=GRIPPER_OPEN_Q,
                         help="从臂全开对应角度 [rad]（未标定，需实测）")
    parser.add_argument("--gripper-close-q", type=float, default=GRIPPER_CLOSE_Q,
                         help="从臂全合对应角度 [rad]（未标定，需实测）")
    parser.add_argument("--gripper-tau-max", type=float, default=GRIPPER_TAU_MAX,
                         help="夹爪最大闭合/张开扭矩 [Nm]，需按电源电流表边测边调")
    parser.add_argument("--gripper-stiffness", type=float, default=GRIPPER_STIFFNESS,
                         help="夹爪虚拟弹簧刚度 [Nm/rad]")
    parser.add_argument("--gripper-kd", type=float, default=GRIPPER_KD,
                         help="夹爪 MIT 阻尼 [Nm·s/rad]")

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
    parser.add_argument("--no-latency-probe", dest="latency_probe",
                         action="store_false", default=LATENCY_PROBE,
                         help="关闭端到端延迟埋点日志")
    parser.add_argument("--latency-probe-threshold", type=float, default=LATENCY_PROBE_THRESHOLD,
                         help="延迟埋点触发/响应阈值（度），越小越灵敏但越容易被噪声误触发")
    parser.add_argument("--no-predict", dest="predict_enabled",
                         action="store_false", default=PREDICT_ENABLED,
                         help="关闭前瞻预测（滤波+弹簧后的速度外推）")
    parser.add_argument("--predict-lookahead-ms", type=float, default=PREDICT_LOOKAHEAD_MS,
                         help="前瞻预测提前量（毫秒）。从小值开始测试，逐步上调，若出现超调/震颤则调小")

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
    if args.no_gripper:
        print(f"  从臂夹爪:      否（--no-gripper）")
    else:
        print(f"  从臂夹爪:      {args.gripper_port}  "
              f"tau_max=±{args.gripper_tau_max:.2f}Nm  stiffness={args.gripper_stiffness:.1f}Nm/rad  "
              f"kd={args.gripper_kd:.2f}")
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
